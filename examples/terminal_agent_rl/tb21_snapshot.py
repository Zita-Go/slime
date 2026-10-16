"""独立评测Pod的CPU模型导出；不加载优化器、不回填缺失的训练权重。"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

from .tb21_queue import atomic_json, sha256


def export_snapshot(checkpoint: Path, origin: Path, output: Path, step: int) -> None:
    import safetensors.torch
    import torch
    import torch.distributed.checkpoint as dist_cp

    from slime.backends.megatron_utils.megatron_to_hf import convert_to_hf

    # GPU镜像的Megatron也有顶层tools包；必须按本仓库绝对文件加载，避免import被遮蔽。
    path = Path(__file__).resolve().parents[2] / "tools/convert_torch_dist_to_hf.py"
    spec = importlib.util.spec_from_file_location("_slime_checkpoint_export", path)
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)

    output.mkdir(parents=True, exist_ok=False)
    # 原生checkpoint可能异步保存；调用方须先确认tracker发布，导出后再复核文件身份未变。
    inputs = {path.name: [path.stat().st_size, path.stat().st_mtime_ns] for path in checkpoint.iterdir() if path.is_file()}
    common = torch.load(checkpoint / "common.pt", map_location="cpu", weights_only=False)
    if common["iteration"] + 1 != step:
        raise ValueError("native checkpoint iteration does not match the requested optimizer update")
    args = common["args"]
    state = {}
    dist_cp.state_dict_loader._load_state_dict(
        state,
        storage_reader=converter.WrappedStorageReader(str(checkpoint)),
        planner=converter.EmptyStateDictLoadPlanner(),
        no_dist=True,
    )
    converted = {}
    for name, tensor in state.items():
        # 固定Megatron把所有专家保存为一个[E,O,I]张量，HF起点也是fused布局。
        # 通用导出器会展开成256个individual expert，不能直接与本模型HF闭包核验。
        name = re.sub(r"\.experts\.experts\.(linear_fc[12])\.weight$", r".experts.\1", name)
        for hf_name, value in convert_to_hf(args, "qwen35moe", f"module.module.{name}", tensor):
            if hf_name in converted:
                raise ValueError(f"duplicate converted parameter: {hf_name}")
            converted[hf_name] = value
    # 在checkpoint转换完成、尚未补vision前检查text权重闭包。
    index = json.loads((origin / "model.safetensors.index.json").read_text())["weight_map"]
    required_text = {name for name in index if not name.startswith("model.visual.")}
    if set(converted) - set(index) or required_text - set(converted):
        raise ValueError(f"converted text weights incomplete: missing={sorted(required_text - set(converted))[:10]}")
    rows = []
    weight_map = {}
    total = 0
    # 按原HF分片布局保存，避免加载整个HF起点的第二份拷贝。
    for filename in sorted(set(index.values())):
        tensors = {}
        with safetensors.safe_open(origin / filename, framework="pt", device="cpu") as source:
            for name in source.keys():
                reference = source.get_slice(name)
                if name in converted:
                    tensor = converted[name]
                    provenance = "checkpoint"
                elif name.startswith("model.visual."):
                    tensor = source.get_tensor(name)
                    provenance = "frozen-origin-vision"
                else:
                    raise ValueError(f"refusing base fallback for trained text tensor: {name}")
                if list(tensor.shape) != reference.get_shape():
                    raise ValueError(f"HF tensor shape mismatch: {name}")
                # 原模型是BF16；不进行降精度转换，dtype必须完全匹配。
                expected_dtype = safetensors.torch._getdtype(reference.get_dtype())
                if tensor.dtype != expected_dtype:
                    raise ValueError(f"HF tensor dtype mismatch: {name}")
                for chunk in tensor.reshape(-1).split(4 * 1024**2):
                    if not torch.isfinite(chunk).all().item():
                        raise ValueError(f"non-finite model tensor: {name}")
                tensors[name] = tensor.contiguous()
                weight_map[name] = filename
                total += tensor.numel() * tensor.element_size()
                rows.append({"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype), "source": provenance})
        safetensors.torch.save_file(tensors, output / filename)
    atomic_json(output / "model.safetensors.index.json", {"metadata": {"total_size": total}, "weight_map": weight_map})
    converter.copy_assets(str(origin), str(output))
    after = {path.name: [path.stat().st_size, path.stat().st_mtime_ns] for path in checkpoint.iterdir() if path.is_file()}
    if after != inputs:
        raise ValueError("checkpoint changed during snapshot export; output retained but not published")
    atomic_json(
        output / "snapshot-receipt.json",
        {
            "ok": True,
            "checkpoint_dir": str(checkpoint),
            "optimizer_step": step,
            "checkpoint_files": inputs,
            "model_tensors": rows,
            "files_sha256": {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()},
            "precision_changed": False,
            "optimizer_loaded": False,
        },
    )
