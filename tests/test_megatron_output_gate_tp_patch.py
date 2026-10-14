"""从固定Megatron源码应用真实Docker补丁，验证GQA gate的TP切分与梯度。"""

import ast
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

NUM_GPUS = 0
PIN = "1dcf0dafa884ad52ffb243625717a3471643e087"
ATTENTION_PATH = "megatron/core/transformer/attention.py"
REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def original_source():
    root = Path(os.environ.get("MEGATRON_ROOT", "/root/Megatron-LM"))
    return subprocess.check_output(["git", "-C", str(root), "show", f"{PIN}:{ATTENTION_PATH}"], text=True)


@pytest.fixture(params=["v0.5.15.post1", "latest"])
def patched_source(request, original_source, tmp_path):
    path = tmp_path / ATTENTION_PATH
    path.parent.mkdir(parents=True)
    path.write_text(original_source)
    patch = REPO_ROOT / "docker" / "patch" / request.param / "megatron.patch"
    subprocess.run(
        ["git", "apply", f"--include={ATTENTION_PATH}", str(patch)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    return path.read_text()


def load_projection(source, rank, gathered_qkv):
    """执行真实projection方法；仅mock TP all-gather，不复制切分实现。"""
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SelfAttention")
    fn = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "get_query_key_value_tensors")
    fn.decorator_list = []
    fn.returns = None
    for arg in fn.args.args:
        arg.annotation = None
    namespace = {
        "torch": torch,
        "Tensor": torch.Tensor,
        "SplitAlongDim": None,
        "apply_module": lambda module: module,
        "get_tensor_model_parallel_rank": lambda: rank,
        "all_gather_last_dim_from_tensor_parallel_region": lambda value: gathered_qkv,
    }
    exec(compile(ast.Module(body=[fn], type_ignores=[]), ATTENTION_PATH, "exec"), namespace)
    return namespace[fn.name]


def project(source, full, tp, rank, output_gate):
    local = full.chunk(tp, dim=-1)[rank]
    obj = SimpleNamespace(
        world_size=tp,
        config=SimpleNamespace(num_query_groups=2, test_mode=False),
        num_attention_heads_per_partition=8,
        num_query_groups_per_partition=1,
        hidden_size_per_attention_head=256,
        q_layernorm=None,
        k_layernorm=None,
        linear_qkv=lambda hidden: (local, None),
    )
    return load_projection(source, rank, full)(obj, None, output_gate=output_gate)


@pytest.mark.unit
@pytest.mark.parametrize("tp", [2, 4, 8])
@pytest.mark.parametrize("output_gate", [False, True])
def test_patch_matches_unsharded_projection_and_gradients(patched_source, tp, output_gate):
    heads_per_group = 18 if output_gate else 10
    for rank in range(tp):
        full = torch.randn(3, 1, 2 * heads_per_group * 256, generator=torch.Generator().manual_seed(123))
        full.requires_grad_()
        result = project(patched_source, full, tp, rank, output_gate)
        query, key, value = result[:3]
        packed = full.reshape(3, 1, 2, heads_per_group, 256)
        begin, end = rank * (16 // tp), (rank + 1) * (16 // tp)
        query_ref = packed[:, :, :, :8, :].reshape(3, 1, 16, 256)[:, :, begin:end, :]
        group = rank // (tp // 2)
        kv_offset = 16 if output_gate else 8
        torch.testing.assert_close(query, query_ref, rtol=0, atol=0)
        torch.testing.assert_close(key, packed[:, :, group : group + 1, kv_offset, :], rtol=0, atol=0)
        torch.testing.assert_close(value, packed[:, :, group : group + 1, kv_offset + 1, :], rtol=0, atol=0)
        if output_gate:
            gate = result[3]
            gate_ref = packed[:, :, :, 8:16, :].reshape(3, 1, 16, 256)[:, :, begin:end, :]
            assert gate.shape == query.shape == (3, 1, 16 // tp, 256)
            torch.testing.assert_close(gate, gate_ref, rtol=0, atol=0)
            # 使用生产gate.view路径，覆盖原有形状错误以及各rank的head选择。
            out = query.reshape(3, 1, -1)
            actual = out * gate.contiguous().view(*out.shape).sigmoid()
            reference = (query_ref * gate_ref.sigmoid()).reshape(3, 1, -1)
        else:
            actual, reference = query, query_ref
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        grad = torch.autograd.grad(actual.sum(), full, retain_graph=True)[0]
        grad_ref = torch.autograd.grad(reference.sum(), full)[0]
        torch.testing.assert_close(grad, grad_ref, rtol=0, atol=0)


@pytest.mark.unit
@pytest.mark.parametrize("tp", [4, 8])
def test_pinned_original_reproduces_invalid_gate_view(original_source, tp):
    full = torch.zeros(3, 1, 2 * 18 * 256)
    for rank in range(tp):
        query, _, _, gate = project(original_source, full, tp, rank, True)
        assert gate.shape[2] == 8
        assert query.shape[2] == 16 // tp
        with pytest.raises(RuntimeError, match="shape"):
            gate.contiguous().view(3, 1, (16 // tp) * 256)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("tp", "pp", "cp", "ep", "etp", "valid"),
    [(2, 1, 32, 32, 1, True), (8, 1, 8, 32, 1, True), (8, 2, 4, 16, 1, True), (8, 1, 16, 32, 1, False), (8, 1, 8, 32, 8, False)],
)
def test_terminal_launcher_matches_megatron_parallel_folding(tp, pp, cp, ep, etp, valid):
    root = Path(os.environ.get("MEGATRON_ROOT", "/root/Megatron-LM"))
    source = subprocess.check_output(["git", "-C", str(root), "show", f"{PIN}:megatron/core/parallel_state.py"], text=True)
    tree = ast.parse(source)
    nodes = [node for node in tree.body if getattr(node, "name", None) in {"RankGenerator", "generate_masked_orthogonal_rank_groups"}]
    namespace = {"List": list}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "parallel_state.py", "exec"), namespace)
    rank_generator = namespace["RankGenerator"]
    if valid:
        dense = rank_generator(tp=tp, ep=1, dp=64 // (tp * pp * cp), pp=pp, cp=cp, order="tp-cp-ep-dp-pp")
        expert = rank_generator(tp=etp, ep=ep, dp=64 // (etp * ep * pp), pp=pp, cp=1, order="tp-cp-ep-dp-pp")
        assert dense.world_size == expert.world_size == 64
        assert dense.get_ranks("pp") == expert.get_ranks("pp")
    script = (REPO_ROOT / "examples/terminal_agent_rl/run_nex_n25_mini.sh").read_text()
    guards = script[script.index("TOTAL_ACTOR_GPUS=") : script.index('\nE2B_VENDOR_DIR="$E2B_VENDOR_DIR" HF_CHECKPOINT=')]
    variables = f"ACTOR_NODES=8 ACTOR_GPUS_PER_NODE=8 TP_SIZE={tp} PP_SIZE={pp} CP_SIZE={cp} EP_SIZE={ep} ETP_SIZE={etp} ROLLOUT_GPUS=32 ROLLOUT_TP_SIZE=2 MAX_CONTEXT_LEN=262144 MAX_PROMPT_LEN=65536 GLOBAL_BATCH_SIZE=128 ROLLOUT_BATCH_SIZE=8 N_SAMPLES_PER_PROMPT=16"
    result = subprocess.run(["bash", "-c", 'die() { echo "$*"; exit 2; };\n' + variables + "\n" + guards], capture_output=True, text=True)
    assert (result.returncode == 0) == valid, result.stdout + result.stderr


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
