# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from qai_hub_models.utils.input_spec import make_torch_inputs

START_TOKEN = 0
END_TOKEN = 1


def generate_path(duration: Tensor, mask: Tensor) -> Tensor:
    """
    Parameters
    ----------
    duration
        shape of [b, 1, t_x], duration time
    mask
        shape of [b, 1, t_y, t_x], attention mask

    Returns
    -------
    attention : Tensor
        the generated self attention
    """
    b, _, t_y, t_x = mask.shape
    cum_duration = torch.cumsum(duration, -1)
    cum_duration_flat = cum_duration.view(b * t_x)
    x = torch.arange(
        t_y, dtype=cum_duration_flat.dtype, device=cum_duration_flat.device
    )
    path = (x.unsqueeze(0) < cum_duration_flat.unsqueeze(1)).to(mask.dtype)
    path = path.view(b, t_x, t_y)

    layer = [[0, 0], [1, 0], [0, 0]][::-1]
    pad_shape = [item for sublist in layer for item in sublist]

    path = path - F.pad(path, pad_shape)[:, :-1]
    return path.unsqueeze(1).transpose(2, 3) * mask


class ByT5Tokenizer:
    def __init__(self) -> None:
        self.special_tokens = {"<pad>": 0, "</s>": 1, "<unk>": 2}
        self.offset = len(self.special_tokens)

    def encode(self, text: str) -> list[int]:
        """
        Parameters
        ----------
        text
            string needed to encode

        Returns
        -------
        list[int]
        """
        return [byte + self.offset for byte in text.encode("utf-8")]

    def decode(self, byte_sequence: list[int]) -> str:
        """
        Parameters
        ----------
        byte_sequence
            byte sequence of encoded text

        Returns
        -------
        str
        """
        byte_sequence = [
            byte - self.offset for byte in byte_sequence if byte >= self.offset
        ]
        return bytes(byte_sequence).decode("utf-8")


def calibrate_charsiu_encoder(
    text: str,
    tokenizer: ByT5Tokenizer,
    inputs: list[list[torch.Tensor | np.ndarray]],
    max_seq_len: int = 50,
) -> None:
    """
    Generate calibration samples for the Charsiu G2P T5 encoder.

    Splits ``text`` into words, prepends each with the ``<eng-us>: `` language
    tag, tokenizes with ByT5, and appends padded (input_ids, attention_mask)
    pairs to ``inputs``. One sample per word.

    Parameters
    ----------
    text
        Input sentence to split into per-word calibration samples.
    tokenizer
        ByT5 byte-level tokenizer instance.
    inputs
        Accumulator list of ``[input_ids_list, attention_mask_list]``.
        New samples are appended in-place.
    max_seq_len
        Sequence length to pad/truncate to (must match the compiled model).
    """
    for word in text.split():
        word = "<eng-us>" + ": " + word

        x = torch.zeros([1, max_seq_len], dtype=torch.int32)
        x_mask = torch.zeros([1, max_seq_len], dtype=torch.int32)
        input_ids = tokenizer.encode(word)
        for idx in range(min(max_seq_len, len(input_ids))):
            x[0, idx] = input_ids[idx]
            x_mask[0, idx] = 1

        inputs[0].append(x)
        inputs[1].append(x_mask)


def calibrate_charsiu_decoder(
    text: str,
    tokenizer: ByT5Tokenizer,
    t5_encoder: Callable[..., list[Tensor]],
    t5_decoder: Callable[..., tuple[Tensor, ...]],
    inputs: list[list[torch.Tensor | np.ndarray]],
    max_seq_len: int = 50,
) -> None:
    """
    Generate calibration samples for the Charsiu G2P T5 decoder.

    Runs the full autoregressive decode loop for a single input sentence:
    encodes ``text`` through ``t5_encoder`` to get cross-attention key/value
    states, then steps ``t5_decoder`` token-by-token until ``END_TOKEN`` or
    ``max_seq_len`` steps. At each step the decoder inputs (input_ids,
    encoder_attention_mask, position, and all past key/value states) are
    appended to ``inputs``.

    Parameters
    ----------
    text
        Input sentence for calibration.
    tokenizer
        ByT5 byte-level tokenizer instance.
    t5_encoder
        Charsiu T5 encoder callable (returns list of cross-attn key/value
        tensors).
    t5_decoder
        Charsiu T5 decoder callable. Must also expose ``.get_input_spec()``
        and ``.block`` (the list of decoder transformer blocks).
    inputs
        Accumulator list with one sub-list per decoder input tensor.
        New samples are appended in-place at each decode step.
    max_seq_len
        Maximum number of decode steps and the padded sequence length for
        encoder inputs (must match the compiled model).
    """
    x = torch.zeros([1, max_seq_len], dtype=torch.int32)
    x_mask = torch.zeros([1, max_seq_len], dtype=torch.int32)

    text = "<eng-us>" + ": " + text
    input_ids = tokenizer.encode(text)
    for idx in range(min(max_seq_len, len(input_ids))):
        x[0, idx] = input_ids[idx]
        x_mask[0, idx] = 1
    enc_out = t5_encoder(x, x_mask)

    dec_dummy_inputs = make_torch_inputs(t5_decoder.get_input_spec())  # type: ignore[attr-defined]
    past_key_values = dec_dummy_inputs[3:]

    assert hasattr(t5_decoder, "block")
    block_num = len(t5_decoder.block)
    for layer_idx in range(block_num):
        past_key_values[block_num * layer_idx + 2] = enc_out[2 * layer_idx]
        past_key_values[block_num * layer_idx + 3] = enc_out[2 * layer_idx + 1]

    token = START_TOKEN
    decoder_input_ids = dec_dummy_inputs[0]
    decoder_input_ids[0, 0] = token
    encoder_attention_mask = x_mask
    for idx in range(max_seq_len):
        inputs[0].append(decoder_input_ids.clone())
        inputs[1].append(encoder_attention_mask.clone())
        inputs[2].append(torch.tensor(idx).reshape(1, 1))
        for i, kv in enumerate(past_key_values):
            inputs[3 + i].append(kv.clone())

        logits, *present_key_values = t5_decoder(
            decoder_input_ids,
            encoder_attention_mask,
            torch.tensor(idx).reshape(1, 1),
            *past_key_values,
        )
        token = int(torch.argmax(logits, dim=-1).item())
        if token == END_TOKEN:
            break

        decoder_input_ids[0, 0] = token
        for layer_idx in range(block_num):
            past_key_values[4 * layer_idx][:, :, idx : idx + 1, :] = present_key_values[
                2 * layer_idx
            ]
            past_key_values[4 * layer_idx + 1][:, :, idx : idx + 1, :] = (
                present_key_values[2 * layer_idx + 1]
            )
