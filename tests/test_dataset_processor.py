"""模型带 processor 时，纯文本题面仍可直接用于自定义 Agent rollout。"""

import json
import sys
import types

import pytest

from slime.utils.data import Dataset

NUM_GPUS = 0


class _Tokenizer:
    def __call__(self, prompts, add_special_tokens=False):
        return {"input_ids": [list(range(len(prompt))) for prompt in prompts]}

    def apply_chat_template(self, *args, **kwargs):
        raise AssertionError("纯文本 Agent 题面不应套用聊天模板")


def _dataset(tmp_path, rows, **kwargs):
    path = tmp_path / "data.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return Dataset(
        str(path),
        tokenizer=_Tokenizer(),
        processor=object(),
        max_length=10,
        prompt_key="prompt",
        label_key="label",
        **kwargs,
    )


def test_raw_text_with_processor_preserves_prompt_and_filters_length(tmp_path, monkeypatch):
    module = types.ModuleType("slime.utils.processing_utils")

    def reject_vision_processing(*args, **kwargs):
        raise AssertionError("纯文本不应调用视觉处理")

    module.process_vision_info = reject_vision_processing
    monkeypatch.setitem(sys.modules, module.__name__, module)
    rows = [
        {"prompt": "题面\n原文", "label": "keep", "metadata": {"item_id": "first"}},
        {"prompt": "x" * 11, "label": "drop"},
        {"prompt": "short", "label": "keep", "metadata": {"item_id": "last"}},
    ]

    dataset = _dataset(tmp_path, rows)

    assert [sample.prompt for sample in dataset.samples] == [rows[0]["prompt"], rows[2]["prompt"]]
    assert [sample.metadata["item_id"] for sample in dataset.samples] == ["first", "last"]
    assert all(sample.label == "keep" and sample.multimodal_inputs is None for sample in dataset.samples)


@pytest.mark.parametrize("raw_prompt", [False, True])
def test_multimodal_messages_still_use_processor(tmp_path, monkeypatch, raw_prompt):
    calls = []
    image = object()
    module = types.ModuleType("slime.utils.processing_utils")

    def process_vision_info(prompt, processor):
        calls.append(prompt)
        return {"images": [image], "videos": None}

    module.process_vision_info = process_vision_info
    monkeypatch.setitem(sys.modules, module.__name__, module)
    content = [{"type": "text", "text": "inspect"}, {"type": "image", "image": "image.png"}]
    row = {
        "prompt": "inspect<image>" if raw_prompt else [{"role": "user", "content": content}],
        "images": ["image.png"],
        "label": "item",
    }

    dataset = _dataset(tmp_path, [row], multimodal_keys={"image": "images"} if raw_prompt else None)

    assert calls == [[{"role": "user", "content": content}]]
    assert dataset[0].prompt == calls[0]
    assert dataset[0].multimodal_inputs == {"images": [image], "videos": None}
