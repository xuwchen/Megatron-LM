# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Qwen ChatML supervision shared by Nemotron training and packing probes."""
from typing import Any

import torch
from PIL import Image


def tokenize(
    processor: Any, messages: list[dict[str, Any]], images: list[Image.Image]
) -> dict[str, torch.Tensor]:
    """Encode a whole conversation with shifted, assistant-only prediction targets."""
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    encoded = processor(
        text=[text], images=images, return_tensors='pt', min_pixels=4096, max_pixels=262144
    )
    tokens = encoded['input_ids'][0]
    ids = tokens.tolist()
    target_mask = torch.zeros_like(tokens, dtype=torch.float32)
    # Qwen3.5's template normalizes thinking blocks, so raw answer text is
    # not necessarily a substring of the rendered conversation. Use ChatML
    # role delimiters from the actual token stream, including reasoning tokens.
    tokenizer = processor.tokenizer
    start_id = int(tokenizer.convert_tokens_to_ids('<|im_start|>'))
    end_id = int(tokenizer.convert_tokens_to_ids('<|im_end|>'))
    header = tokenizer.encode('assistant\n', add_special_tokens=False)
    if not header or start_id == end_id:
        raise ValueError('The processor does not provide the expected ChatML boundaries')
    marked_turns = 0
    for i, token in enumerate(ids):
        if token != start_id or ids[i + 1 : i + 1 + len(header)] != header:
            continue
        content_start = i + 1 + len(header)
        try:
            content_end = ids.index(end_id, content_start)
        except ValueError as exc:
            raise ValueError('An assistant message has no closing ChatML delimiter') from exc
        target_mask[content_start:content_end] = 1
        marked_turns += 1
    expected_turns = sum(turn['role'] == 'assistant' for turn in messages)
    if marked_turns != expected_turns:
        raise ValueError(f'Assistant turn count mismatch: {marked_turns} != {expected_turns}')
    labels = torch.cat([tokens[1:], torch.tensor([-100], dtype=torch.long)])
    loss_mask = torch.cat([target_mask[1:], torch.zeros(1)])
    special = torch.tensor(processor.tokenizer.all_special_ids, dtype=torch.long)
    loss_mask[torch.isin(labels, special)] = 0
    labels[loss_mask == 0] = -100
    if not bool(loss_mask.any()):
        raise ValueError('Sample has no assistant prediction targets')
    return {
        'input_ids': tokens,
        'labels': labels,
        'loss_mask': loss_mask,
        'pixel_values': encoded['pixel_values'].to(torch.bfloat16),
        'image_grid_thw': encoded['image_grid_thw'],
    }
