# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from functools import partial
from typing import Any

import torch
from PIL import Image
from transformers import (
    AutoModel,
    AutoProcessor,
    GenerationConfig,
    set_seed,
)

from qai_hub_models import Precision
from qai_hub_models.models._shared.llm.app import IndentedTextStreamer
from qai_hub_models.models._shared.llm.generator_factory import make_generator
from qai_hub_models.models._shared.llm.model import (
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_EXPORT_SEQUENCE_LENGTHS,
    LLM_QNN,
    LLM_AIMETOnnx,
    LLMBase,
    SplitForwardMixin,
    get_tokenizer,
)
from qai_hub_models.models._shared.lm_driver.generator import VLM_Generator
from qai_hub_models.utils.args import add_input_spec_args, get_model_cli_parser
from qai_hub_models.utils.checkpoint import (
    CheckpointSpec,
    CheckpointType,
)
from qai_hub_models.utils.huggingface import has_model_access

# Max output tokens to generate
# You can override this with cli argument.
MAX_OUTPUT_TOKENS = 1000


def llm_chat_demo(
    model_cls: type[LLM_AIMETOnnx],
    fp_model_cls: type[LLMBase],
    qnn_model_cls: type[LLM_QNN],
    model_id: str,
    end_tokens: set[str],
    hf_repo_name: str,
    hf_repo_url: str,
    default_prompt: str | None = None,
    raw: bool = False,
    test_checkpoint: CheckpointSpec | None = None,
    supports_thinking: bool = False,
    enable_thinking: bool | None = None,
    # VLM parameters (optional)
    vision_encoder_cls: Any | None = None,
    default_sequence_length: list[int] | int | None = None,
) -> None:
    """Shared Chat Demo App to generate output for provided input prompt.

    When enable_thinking is not None it overrides the --thinking / --no-thinking
    CLI flag. Tests pass enable_thinking=False so small thinking models answer
    directly instead of looping in an unterminated reasoning trace.
    """
    parser = get_model_cli_parser(
        model_cls,
        suppress_help_arguments=["--host-device", "--fp-model", "--precision"],
    )
    parser = add_input_spec_args(model_cls, parser)

    if default_sequence_length is None:
        default_sequence_length = DEFAULT_EXPORT_SEQUENCE_LENGTHS
    elif isinstance(default_sequence_length, int):
        default_sequence_length = [default_sequence_length]

    parser.add_argument(
        "--sequence-length",
        dest="sequence_length",
        type=int,
        nargs="+",
        default=sorted(default_sequence_length),
        help="One or more AR sequence-length buckets. With multiple values the "
        "generator selects the smallest bucket that fits each step",
    )
    parser.add_argument("--prompt", type=str, default=None, help="input prompt.")
    parser.add_argument(
        "--prompt-file", type=str, default=None, help="input prompt from file path."
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="If specified, will assume prompt contains system tags and will not be added automatically.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=MAX_OUTPUT_TOKENS,
        help="max output tokens to generate.",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=DEFAULT_CONTEXT_LENGTH,
        help="Context length (total KV cache size).",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed.")

    if vision_encoder_cls is not None:
        parser.add_argument(
            "--image",
            type=str,
            nargs="+",
            default=None,
            help="Path(s) to input image(s). Pass multiple paths for multi-image prompts.",
        )
        parser.add_argument(
            "--image-height",
            type=int,
            default=None,
            help="VEG image height (must be divisible by patch_size * spatial_merge_size).",
        )
        parser.add_argument(
            "--image-width",
            type=int,
            default=None,
            help="VEG image width (must be divisible by patch_size * spatial_merge_size).",
        )

    if supports_thinking:
        parser.add_argument(
            "--thinking",
            action="store_true",
            dest="thinking",
            default=True,
            help="Enable thinking mode (default).",
        )
        parser.add_argument(
            "--no-thinking",
            action="store_false",
            dest="thinking",
            help="Disable thinking mode by adding empty thinking tags.",
        )

    args = parser.parse_args([] if test_checkpoint is not None else None)
    checkpoint = args.checkpoint if test_checkpoint is None else test_checkpoint
    max_output_tokens = args.max_output_tokens if test_checkpoint is None else 1000

    if args.prompt is not None and args.prompt_file is not None:
        raise ValueError("Must specify one of --prompt or --prompt-file")
    if args.prompt_file is not None:
        with open(args.prompt_file) as f:
            prompt = f.read()
    elif args.prompt:
        prompt = args.prompt
    elif default_prompt is not None:
        prompt = default_prompt
    else:
        prompt = fp_model_cls.default_user_prompt

    # Allow literal "\n" in prompts to be interpreted as newlines.
    prompt = prompt.replace("\\n", "\n")

    assert checkpoint is not None
    checkpoint_type = CheckpointType.from_checkpoint(checkpoint)
    is_default = isinstance(checkpoint, str) and checkpoint.startswith("DEFAULT")
    if not is_default:
        tokenizer = get_tokenizer(checkpoint)
    else:
        has_model_access(hf_repo_name, hf_repo_url)
        tokenizer = get_tokenizer(hf_repo_name)

    # Build the prompt formatting function
    if args.raw or raw:

        def preprocess_prompt_fn(
            user_input_prompt: str = "",
            system_context_prompt: str = "",
        ) -> str:
            return user_input_prompt
    elif supports_thinking:
        thinking = enable_thinking if enable_thinking is not None else args.thinking
        preprocess_prompt_fn = partial(
            fp_model_cls.get_input_prompt_with_tags,
            tokenizer=tokenizer,
            enable_thinking=thinking,
        )
    else:
        preprocess_prompt_fn = partial(
            fp_model_cls.get_input_prompt_with_tags, tokenizer=tokenizer
        )

    # Get image path if VLM
    image_path = (
        getattr(args, "image", None) if vision_encoder_cls is not None else None
    )

    if test_checkpoint is None:
        print(f"\n{'-' * 85}")
        print(f"** Generating response via {model_id} **")
        if checkpoint_type == CheckpointType.GENIE_BUNDLE:
            print("Variant: ON-DEVICE (QNN)")
            print("    This runs on the target hardware.")
        elif checkpoint_type.is_aimet_onnx():
            print("Variant: QUANTIZED (AIMET-ONNX)")
            print("    This aims to replicate on-device accuracy through simulation.")
        elif issubclass(fp_model_cls, SplitForwardMixin):
            print("Variant: UNQUANTIZED (ONNX Split)")
        else:
            print("Variant: FLOATING POINT (PyTorch)")
            print("    This runs the original unquantized model for baseline purposes.")
        print()
        print("Prompt:", prompt)
        if vision_encoder_cls is not None:
            print("Image:", image_path if image_path else "(none - text only)")
        print("Raw (prompt will be passed in unchanged):", args.raw)
        if supports_thinking:
            print("Thinking mode:", "enabled" if args.thinking else "disabled")
        print("Max number of output tokens to generate:", args.max_output_tokens)
        print()
        print(f"{'-' * 85}\n")

    set_seed(args.seed)
    host_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    final_model_cls: type[LLMBase | LLM_AIMETOnnx | LLM_QNN]
    extra: dict[str, Any] = {}

    if checkpoint_type == CheckpointType.GENIE_BUNDLE:
        final_model_cls = qnn_model_cls
        is_fp = False
    elif checkpoint_type.is_aimet_onnx():
        # VLMs need the FP model so the quantized instance inherits
        # _original_llm_config (used by make_generator) — the quantized
        # checkpoint alone doesn't carry it. Text LLMs only need it to seed a
        # DEFAULT (non-UNQUANTIZED) checkpoint's tokenizer/config.
        needs_fp_model = vision_encoder_cls is not None or (
            is_default and checkpoint != "DEFAULT_UNQUANTIZED"
        )
        if needs_fp_model:
            extra["fp_model"] = fp_model_cls.from_pretrained()
        final_model_cls = model_cls
        is_fp = False
    else:
        final_model_cls = fp_model_cls
        is_fp = True

    # Collect VLM image size override if provided
    vlm_image_size = None
    if vision_encoder_cls is not None:
        img_h = getattr(args, "image_height", None)
        img_w = getattr(args, "image_width", None)
        if img_h is not None and img_w is not None:
            vlm_image_size = (img_w, img_h)

    model_params: dict[str, Any] = {
        "host_device": host_device,
        **extra,
    }
    checkpoint_for_model = None if checkpoint == "DEFAULT_UNQUANTIZED" else checkpoint
    if checkpoint_for_model is not None:
        model_params["checkpoint"] = checkpoint_for_model

    model = final_model_cls.from_pretrained(**model_params).to(host_device)

    vision_model = None
    if (
        issubclass(fp_model_cls.GeneratorClass, VLM_Generator)  # type: ignore[attr-defined]
        and vision_encoder_cls is not None
        and image_path
    ):
        if is_fp:
            veg_precision = Precision.float
        else:
            default_precision = getattr(model_cls, "default_precision", Precision.w4a16)
            veg_precision = checkpoint_type.precision(
                default_precision, checkpoint=checkpoint
            )

        if veg_precision == Precision.float:
            hf_model = AutoModel.from_pretrained(hf_repo_name, trust_remote_code=True)
            vision_model = hf_model.visual.to(host_device).eval()
            del hf_model
        else:
            veg_kwargs: dict[str, Any] = {
                "checkpoint": checkpoint,
                "precision": veg_precision,
            }
            if vlm_image_size is not None:
                veg_kwargs["image_width"] = vlm_image_size[0]
                veg_kwargs["image_height"] = vlm_image_size[1]
            vision_model = vision_encoder_cls.from_pretrained(
                device=host_device,
                **veg_kwargs,
            )

    generator = make_generator(
        model,
        sequence_length=args.sequence_length,
        context_length=args.context_length,
        vision_model=vision_model,
        model_cls=fp_model_cls,
    )

    end_token_ids = []
    for token in end_tokens:
        token_ids = tokenizer.encode(token, add_special_tokens=False)
        if len(token_ids) == 1:
            end_token_ids.append(token_ids[0])
    end_token_ids.append(tokenizer.eos_token_id)

    generator.generation_config = GenerationConfig(  # type: ignore[assignment, unused-ignore]
        max_new_tokens=max_output_tokens,
        eos_token_id=end_token_ids,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True,
        top_k=40,
        top_p=0.95,
        temperature=0.8,
    )

    streamer = IndentedTextStreamer(
        tokenizer=tokenizer,
        skip_prompt=False,
        line_start="    + ",
    )

    input_prompt_processed = preprocess_prompt_fn(user_input_prompt=prompt)
    input_tokens = tokenizer(
        input_prompt_processed,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(host_device)

    generate_kwargs: dict[str, Any] = {
        "inputs": input_tokens["input_ids"],
        "attention_mask": input_tokens["attention_mask"],
        "generation_config": generator.generation_config,
        "streamer": streamer,
    }

    # For VLMs with images, resize to VEG dimensions and preprocess
    if vision_model is not None and image_path:
        processor = AutoProcessor.from_pretrained(hf_repo_name)

        if hasattr(vision_model, "veg"):
            veg = vision_model.veg
            target_w = int(veg._image_width)
            target_h = int(veg._image_height)
        elif hasattr(vision_model, "_image_width") and hasattr(
            vision_model, "_image_height"
        ):
            # Fixed-shape VEG (e.g. Qwen2.5-VL): images must match the shape the
            # encoder was exported/quantized for.
            target_w = int(vision_model._image_width)
            target_h = int(vision_model._image_height)
        elif vlm_image_size is not None:
            target_w, target_h = vlm_image_size
        elif vision_encoder_cls is not None and hasattr(
            vision_encoder_cls, "DEFAULT_IMAGE_SIZE"
        ):
            target_h, target_w = vision_encoder_cls.DEFAULT_IMAGE_SIZE
        else:
            target_w, target_h = 512, 512
        images = [
            Image.open(p).convert("RGB").resize((target_w, target_h))
            for p in image_path
        ]

        formatted_text = fp_model_cls.get_input_prompt_with_tags(
            user_input_prompt=prompt,
            include_image=True,
            tokenizer=tokenizer,
        )
        processed = processor(
            text=[formatted_text],
            images=images,
            return_tensors="pt",
        ).to(host_device)

        generate_kwargs["inputs"] = processed["input_ids"]
        generate_kwargs["attention_mask"] = processed["attention_mask"]
        if "pixel_values" in processed:
            generate_kwargs["pixel_values"] = processed["pixel_values"]
        if "image_grid_thw" in processed:
            generate_kwargs["image_grid_thw"] = processed["image_grid_thw"]

    generator.generate(**generate_kwargs)  # type: ignore[operator, misc, unused-ignore]
    generator.model.cleanup()
