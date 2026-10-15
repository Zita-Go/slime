"""回移已知gate补丁，或用torchrun验证真实Megatron/TE gated attention。

在空闲GPU Pod内先用 --apply-only，再分别以4/8个rank运行；不操作Ray生命周期。
"""

import argparse
import hashlib
import inspect
import json
import os
import subprocess
from datetime import timedelta
from pathlib import Path


PIN = "1dcf0dafa884ad52ffb243625717a3471643e087"
ATTENTION_PATH = "megatron/core/transformer/attention.py"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply-only", action="store_true")
    parser.add_argument("--megatron-root", type=Path, default=Path("/root/Megatron-LM"))
    parser.add_argument("--tp-size", type=int, choices=[2, 4, 8])
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    source = args.megatron_root / ATTENTION_PATH
    if args.apply_only:
        commit = subprocess.check_output(["git", "-C", str(args.megatron_root), "rev-parse", "HEAD"], text=True).strip()
        if commit != PIN:
            raise RuntimeError(f"拒绝对未知Megatron版本应用补丁: {commit}")
        patch = Path(__file__).resolve().parents[1] / "docker/patch/v0.5.15.post1/megatron.patch"
        command = ["git", "-C", str(args.megatron_root), "apply", f"--include={ATTENTION_PATH}"]
        present = subprocess.run([*command, "--reverse", "--check", str(patch)], capture_output=True).returncode == 0
        if not present:
            subprocess.run([*command, "--check", str(patch)], check=True)
            subprocess.run([*command, str(patch)], check=True)
        print(json.dumps({"ok": True, "already_applied": present, "commit": commit, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}))
        return

    if args.tp_size is None or args.receipt is None:
        parser.error("GPU验证需要 --tp-size 和一个新的 --receipt 路径")
    if args.receipt.exists():
        raise FileExistsError(f"保留已有回执: {args.receipt}")
    import torch
    import torch.distributed as dist
    from megatron.core import parallel_state
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer import TransformerConfig
    from megatron.core.transformer.attention import SelfAttention
    from megatron.core.transformer.enums import AttnMaskType

    if "gate = gate[:, :, idx * size : (idx + 1) * size, :]" not in source.read_text():
        raise RuntimeError("GPU Pod仍使用没有gate切分修复的Megatron")
    if Path(inspect.getfile(SelfAttention)).resolve() != source.resolve():
        raise RuntimeError("实际import的Megatron与指定源码路径不同")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(minutes=5))
    rank = dist.get_rank()
    assert dist.get_world_size() == args.tp_size
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=args.tp_size, pipeline_model_parallel_size=1)
    reports = []
    try:
        for sp in [False, True]:
            model_parallel_cuda_manual_seed(123)
            config = TransformerConfig(
                tensor_model_parallel_size=args.tp_size,
                num_layers=1,
                hidden_size=2048,
                num_attention_heads=16,
                num_query_groups=2,
                kv_channels=256,
                normalization="RMSNorm",
                bf16=True,
                params_dtype=torch.bfloat16,
                use_cpu_initialization=True,
                attention_output_gate=True,
                qk_layernorm=True,
                add_bias_linear=False,
                hidden_dropout=0.0,
                attention_dropout=0.0,
                gradient_accumulation_fusion=False,
                sequence_parallel=sp,
            )
            spec = get_gpt_layer_with_transformer_engine_spec(qk_layernorm=True).submodules.self_attention.submodules
            attention = SelfAttention(config, spec, layer_number=1, attn_mask_type=AttnMaskType.causal).cuda()
            torch.manual_seed(123)
            local_seq = 64 // args.tp_size if sp else 64
            hidden = torch.randn(local_seq, 2, 2048, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            query, _, _, gate = attention.get_query_key_value_tensors(hidden, output_gate=True)
            assert query.shape == gate.shape == (64, 2, 16 // args.tp_size, 256), (query.shape, gate.shape)
            del query, gate
            output, bias = attention(hidden, attention_mask=None)
            assert output.shape == hidden.shape, output.shape
            if bias is not None:
                output = output + bias
            assert torch.isfinite(output).all()
            loss = output.float().square().mean()
            loss.backward()
            assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
            gradients = [p.grad for p in attention.parameters() if p.grad is not None]
            assert gradients and all(torch.isfinite(g).all() for g in gradients)
            norm = sum(g.float().square().sum().item() for g in gradients) ** 0.5
            assert norm > 0
            reports.append({"rank": rank, "tp": args.tp_size, "sequence_parallel": sp, "loss": loss.item(), "grad_norm": norm, "peak_allocated_bytes": torch.cuda.max_memory_allocated()})
            del attention, hidden, output, loss, gradients
            torch.cuda.empty_cache()
        gathered = [None] * args.tp_size
        dist.all_gather_object(gathered, reports)
        if rank == 0:
            args.receipt.parent.mkdir(parents=True, exist_ok=True)
            with args.receipt.open("x") as handle:
                json.dump({"ok": True, "scope": "single_gated_attention_forward_backward_only", "full_model_rl_verified": False, "tp": args.tp_size, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "ranks": gathered}, handle, indent=2)
                handle.write("\n")
            print(json.dumps({"ok": True, "tp": args.tp_size, "receipt": str(args.receipt)}))
        dist.barrier()
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
