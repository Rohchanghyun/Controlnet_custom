#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import functools
import gc
import re
import logging
import math
import os
import random
import shutil
from contextlib import nullcontext
from pathlib import Path
import json

import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, ProjectConfiguration, set_seed
from datasets import load_dataset, Dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UniPCMultistepScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available, make_image_grid
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_torch_npu_available, is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

from safetensors.torch import load_file, save_file
from pipeline.pipeline_controlnext import StableDiffusionXLControlNeXtPipeline
from models.controlnet import ControlNetModel
from models.unet import UNet2DConditionModel

import sys
from network import CLIP_Attention_Image_adapter_SDXL, MGN
from transformers import CLIPProcessor, CLIPModel

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.31.0.dev0")

logger = get_logger(__name__)
if is_torch_npu_available():
    torch.npu.config.allow_internal_format = False


def log_validation(vae, unet, controlnet, args, accelerator, weight_dtype, step, is_final_validation=False, guidance_scale=7.5):
    logger.info(f"Running validation with guidance_scale={guidance_scale}... ")

    pipeline = StableDiffusionXLControlNeXtPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        unet=unet,
        controlnet=controlnet,
        safety_checker=None,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )

    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    if len(args.validation_image) == len(args.validation_prompt):
        validation_images = args.validation_image
        validation_prompts = args.validation_prompt
    elif len(args.validation_image) == 1:
        validation_images = args.validation_image * len(args.validation_prompt)
        validation_prompts = args.validation_prompt
    elif len(args.validation_prompt) == 1:
        validation_images = args.validation_image
        validation_prompts = args.validation_prompt * len(args.validation_image)
    else:
        raise ValueError(
            "number of `args.validation_image` and `args.validation_prompt` should be checked in `parse_args`"
        )

    image_logs = []
    if is_final_validation or torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(accelerator.device.type)

    for validation_prompt, validation_image in zip(validation_prompts, validation_images):
        validation_image = Image.open(validation_image).convert("RGB")
        validation_image = validation_image.resize((args.resolution, args.resolution))

        images = []

        for _ in range(args.num_validation_images):
            with autocast_ctx:
                image = pipeline(
                    prompt=validation_prompt,
                    controlnet_image=validation_image,
                    controlnet_scale_factor=args.controlnet_scale_factor,
                    guidance_scale=guidance_scale,
                    num_inference_steps=20,
                    generator=generator,
                    width=args.resolution,
                    height=args.resolution,
                ).images[0]
            images.append(image)

        image_logs.append(
            {"validation_image": validation_image, "images": images, "validation_prompt": validation_prompt}
        )

    sample_dir = os.path.join(args.output_dir, "samples", f"sample-{step}")
    os.makedirs(sample_dir, exist_ok=True)
    tracker_key = "test" if is_final_validation else "validation"
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                validation_image = log["validation_image"]

                formatted_images = []

                formatted_images.append(np.asarray(validation_image))

                for image in images:
                    formatted_images.append(np.asarray(image))

                formatted_images = np.stack(formatted_images)

                tracker.writer.add_images(validation_prompt, formatted_images, step, dataformats="NHWC")
        elif tracker.name == "wandb":
            formatted_images = []

            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                validation_image = log["validation_image"]

                formatted_images.append(wandb.Image(validation_image, caption="Controlnet conditioning"))

                for image in images:
                    image = wandb.Image(image, caption=validation_prompt)
                    formatted_images.append(image)

            tracker.log({tracker_key: formatted_images})
        else:
            logger.warning(f"image logging not implemented for {tracker.name}")

    formatted_images = []
    formatted_images.append(validation_image)
    for i, image in enumerate(images):
        formatted_images.append(image)
        image.save(os.path.join(sample_dir, f"image-{i}_{step}.png"))
    image_grid = make_image_grid(formatted_images, 1, len(formatted_images))
    image_grid.save(os.path.join(sample_dir, f"grid_{step}.png"))
    logger.info(f"{len(formatted_images)} validation images saved to {sample_dir}")

    del pipeline
    gc.collect()
    torch.cuda.empty_cache()

    return image_logs


def save_models(unet, controlnet, output_dir, args, orig_unet_sd=None):
    os.makedirs(output_dir, exist_ok=True)
    unet_sd = unet.state_dict()
    pattern = re.compile(args.unet_trainable_param_pattern)
    unet_sd = {k: v for k, v in unet_sd.items() if pattern.match(k)}
    if args.save_weights_increaments:
        for k, v in unet_sd.items():
            unet_sd[k] = unet_sd[k].detach().cpu() - orig_unet_sd[k]
    save_file(unet_sd, os.path.join(output_dir, "unet_weight_increasements.safetensors"))
    save_file(controlnet.state_dict(), os.path.join(output_dir, "controlnet.safetensors"))


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


def save_model_card(repo_id: str, image_logs=None, base_model=str, repo_folder=None):
    img_str = ""
    if image_logs is not None:
        img_str = "You can find some example images below.\n\n"
        for i, log in enumerate(image_logs):
            images = log["images"]
            validation_prompt = log["validation_prompt"]
            validation_image = log["validation_image"]
            validation_image.save(os.path.join(repo_folder, "image_control.png"))
            img_str += f"prompt: {validation_prompt}\n"
            images = [validation_image] + images
            make_image_grid(images, 1, len(images)).save(os.path.join(repo_folder, f"images_{i}.png"))
            img_str += f"![images_{i})](./images_{i}.png)\n"

    model_description = f"""
# controlnet-{repo_id}

These are controlnet weights trained on {base_model} with new type of conditioning.
{img_str}
"""

    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="openrail++",
        base_model=base_model,
        model_description=model_description,
        inference=True,
    )

    tags = [
        "stable-diffusion-xl",
        "stable-diffusion-xl-diffusers",
        "text-to-image",
        "diffusers",
        "controlnet",
        "diffusers-training",
    ]
    model_card = populate_model_card(model_card, tags=tags)

    model_card.save(os.path.join(repo_folder, "README.md"))


class LossRecorder:
    r"""
    Class to record better losses.
    """

    def __init__(self, gamma=0.9, max_window=None):
        self.losses = []
        self.gamma = gamma
        self.ema = 0
        self.t = 0
        self.max_window = max_window

    def add(self, *, loss: float) -> None:
        self.losses.append(loss)
        if self.max_window is not None and len(self.losses) > self.max_window:
            self.losses.pop(0)
        self.t += 1
        ema = self.ema * self.gamma + loss * (1 - self.gamma)
        ema_hat = ema / (1 - self.gamma ** self.t) if self.t < 500 else ema
        self.ema = ema_hat

    def moving_average(self, *, window: int) -> float:
        if len(self.losses) < window:
            window = len(self.losses)
        return sum(self.losses[-window:]) / window


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to an improved VAE to stabilize training. For more details check out: https://github.com/huggingface/diffusers/pull/4038.",
    )
    parser.add_argument(
        "--pretrained_unet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained unet safetensors file if you want to continue training.",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained controlnet safetensors file if you want to continue training.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--use_safetensors",
        action="store_true",
        help="Whether or not to set use_safetensors=True for loading the pretrained model.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="controlnet-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--controlnet_scale_factor",
        type=float,
        default=1.0,
        help=(
            "The scale factor for the controlnet. This is used to scale the controlnet output before adding it to the unet output."
            " For depth control, we recommend setting this to 1.0."
            " For canny control, we recommend setting this to 0.35."
        )
    )
    parser.add_argument(
        "--save_weights_increaments",
        type=bool,
        default=False,
        help="Save the unet weights in increaments",
    )
    parser.add_argument(
        "--load_weights_increaments",
        type=bool,
        default=False,
        help="Load the unet weights in increaments",
    )
    parser.add_argument(
        "--crops_coords_top_left_h",
        type=int,
        default=0,
        help=("Coordinate for (the height) to be included in the crop coordinate embeddings needed by SDXL UNet."),
    )
    parser.add_argument(
        "--crops_coords_top_left_w",
        type=int,
        default=0,
        help=("Coordinate for (the height) to be included in the crop coordinate embeddings needed by SDXL UNet."),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=25)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--unet_trainable_param_pattern",
        type=str,
        default=r".*attn2.*to_out.*",
        help="Regex pattern to match the name of trainable parameters of the UNet.",
    )
    parser.add_argument(
        "--learning_rate_controlnet",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use for the controlnet.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant_with_warmup",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--optimizer_type",
        type=str,
        default="adafactor",
        help="The optimizer type to use. Choose between ['adamw', 'adafactor']",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--adafactor_relative_step", type=bool, default=False, help="Relative step size for Adafactor.")
    parser.add_argument("--adafactor_scale_parameter", type=bool, default=False, help="Scale the initial parameter.")
    parser.add_argument("--adafactor_warmup_init", type=bool, default=False, help="Warmup the initial parameter.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--enable_npu_flash_attention", action="store_true", help="Whether or not to use npu flash attention."
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing the target image."
    )
    parser.add_argument(
        "--conditioning_image_column",
        type=str,
        default="conditioning_image",
        help="The column of the dataset containing the controlnet conditioning image.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of paths to the controlnet conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--validation_source_image",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of paths to the original images that will be used for feature extraction with the image adapter"
            " during validation. These should correspond to the sketch images in --validation_image."
            " If not provided, the sketch images themselves will be used for feature extraction."
        ),
    )
    parser.add_argument(
        "--validation_guidance_scale",
        type=float,
        default=7.5,
        help="The guidance scale (CFG scale) to use for validation. Higher values give stronger text guidance.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="sd_xl_train_controlnet",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--tracker_run_name",
        type=str,
        default=None,
        help=(
            "The `run_name` argument passed to Accelerator.init_trackers for wandb logging."
            " If not provided, a timestamp will be used as default run name."
        ),
    )
    parser.add_argument(
        "--extractor_weight",
        type=str,
        default=None,
        help="Path to MGN extractor weights",
    )
    parser.add_argument(
        "--image_adapter_weight",
        type=str,
        default=None,
        help="Path to Image Adapter weights",
    )
    parser.add_argument(
        "--id_images_path",
        type=str,
        default="projects/emoji_generation/Controlnet_custom/id_images",
        help="Path to ID images for Image Adapter",
    )
    parser.add_argument(
        "--id_captions_path",
        type=str,
        default="projects/emoji_generation/Controlnet_custom/id_captions.txt",
        help="Path to ID captions for Image Adapter",
    )
    parser.add_argument(
        "--control_blend_weight",
        type=float,
        default=0.7,
        help="Weight for blending original control with ID control",
    )
    parser.add_argument(
        "--id_control_blend_weight",
        type=float,
        default=0.3,
        help="Weight for blending ID control with original control",
    )
    parser.add_argument(
        "--clip_embeddings_dim",
        type=int,
        default=8,
        help="Dimension for the CLIP image adapter embeddings sequence length",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Specify either `--dataset_name` or `--train_data_dir`")

    if args.dataset_name is not None and args.train_data_dir is not None:
        raise ValueError("Specify only one of `--dataset_name` or `--train_data_dir`")

    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")

    if args.validation_prompt is not None and args.validation_image is None:
        raise ValueError("`--validation_image` must be set if `--validation_prompt` is set")

    if args.validation_prompt is None and args.validation_image is not None:
        raise ValueError("`--validation_prompt` must be set if `--validation_image` is set")

    if (
        args.validation_image is not None
        and args.validation_prompt is not None
        and len(args.validation_image) != 1
        and len(args.validation_prompt) != 1
        and len(args.validation_image) != len(args.validation_prompt)
    ):
        raise ValueError(
            "Must provide either 1 `--validation_image`, 1 `--validation_prompt`,"
            " or the same number of `--validation_prompt`s and `--validation_image`s"
        )
        
    # Add validation for validation_source_image
    if (
        args.validation_source_image is not None
        and args.validation_image is not None
        and len(args.validation_source_image) != 1
        and len(args.validation_image) != 1
        and len(args.validation_source_image) != len(args.validation_image)
    ):
        raise ValueError(
            "Must provide either 1 `--validation_source_image`, 1 `--validation_image`,"
            " or the same number of `--validation_source_image`s and `--validation_image`s"
        )

    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder."
        )

    return args


def get_train_dataset(args, accelerator):
    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    while True:
        try:
            if args.dataset_name is not None:
                # Downloading and loading a dataset from the hub.
                print(f"args.dataset_name: {args.dataset_name}")
                dataset = load_dataset(
                    args.dataset_name,
                    args.dataset_config_name,
                    cache_dir=args.cache_dir,
                )
            else:
                if args.train_data_dir is not None:
                    # 1. 메타데이터 직접 로드
                    metadata_path = Path(args.train_data_dir) / "metadata.jsonl"
                    with open(metadata_path, 'r') as f:
                        items = [json.loads(line) for line in f]
                    
                    # 2. 데이터 검증 및 경로 변환
                    valid_items = []
                    for item in items:
                        img_path = Path(item['file_name'])
                        cond_path = Path(item['conditioning_image'])
                        
                        # 상대 경로 처리
                        if not img_path.is_absolute():
                            img_path = Path(args.train_data_dir) / img_path
                        if not cond_path.is_absolute():
                            cond_path = Path(args.train_data_dir) / cond_path
                        
                        if img_path.exists() and cond_path.exists():
                            valid_items.append({
                                'image': str(img_path),
                                'conditioning_image': str(cond_path),
                                'text': item['text']
                            })
                        else:
                            logger.error(f"🚨 Missing files: {img_path} | {cond_path}")
                    
                    # 3. Dataset 생성 (HuggingFace Dataset 올바른 생성 방식)
                    from datasets import Dataset
                    # valid_items의 키를 직접 매핑
                    dataset_dict = {
                        args.image_column: [x['image'] for x in valid_items],
                        args.conditioning_image_column: [x['conditioning_image'] for x in valid_items],
                        args.caption_column: [x['text'] for x in valid_items]
                    }
                    dataset = Dataset.from_dict(dataset_dict)
                    logger.info(f"✅ Dataset structure: {dataset}")
            break
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            logger.error("Retry...")
            continue

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset.column_names

    # 6. Get the column names for input/target.
    if args.image_column is None:
        image_column = column_names[0]
        logger.info(f"image column defaulting to {image_column}")
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"`--image_column` value '{args.image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    if args.caption_column is None:
        caption_column = column_names[1]
        logger.info(f"caption column defaulting to {caption_column}")
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"`--caption_column` value '{args.caption_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    if args.conditioning_image_column is None:
        conditioning_image_column = column_names[2]
        logger.info(f"conditioning image column defaulting to {conditioning_image_column}")
    else:
        conditioning_image_column = args.conditioning_image_column
        if conditioning_image_column not in column_names:
            raise ValueError(
                f"`--conditioning_image_column` value '{args.conditioning_image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    with accelerator.main_process_first():
        train_dataset = dataset.shuffle(seed=args.seed)
        if args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(args.max_train_samples))
    return train_dataset


# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt(prompt_batch, text_encoders, tokenizers, proportion_empty_prompts, is_train=True):
    prompt_embeds_list = []

    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    with torch.no_grad():
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer(
                captions,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            prompt_embeds = text_encoder(
                text_input_ids.to(text_encoder.device),
                output_hidden_states=True,
            )

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds

# 칠해진 부분을 검정색으로 변환하는 함수
def process_sketch(x):
    # 배경은 흰색(1.0)이라고 가정
    # 어떤 색이든 칠해진 부분(배경보다 어두운 부분)은 검정색(0.0)으로 변환
    threshold = 0.95  # 이 값은 배경과 칠해진 부분을 구분하는 임계값
    
    # 모든 채널에서 threshold보다 작은 값(칠해진 부분)을 0으로 설정
    x_processed = torch.where(x < threshold, torch.zeros_like(x), torch.ones_like(x))
    
    return 1-x_processed

def prepare_train_dataset(dataset, accelerator):
    train_transforms = transforms.Compose(
        [
            transforms.Resize((args.resolution, args.resolution), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    # 스케치(conditioning) 이미지용 transform도 동일하게 수정
    train_conditioning_transforms = transforms.Compose(
        [
            transforms.Resize((args.resolution, args.resolution), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def preprocess_train(examples):
        # 이미지 파일 경로에서 실제 이미지 로드
        images = [Image.open(image_path).convert("RGB") for image_path in examples[args.image_column]]
        images = [train_transforms(image) for image in images]

        # 컨디셔닝 이미지 처리
        conditioning_images = [Image.open(cond_path).convert("RGB") for cond_path in examples[args.conditioning_image_column]]
        conditioning_images = [train_conditioning_transforms(image) for image in conditioning_images]

        examples["pixel_values"] = images
        examples["conditioning_pixel_values"] = conditioning_images

        return examples

    with accelerator.main_process_first():
        dataset = dataset.with_transform(preprocess_train)

    return dataset


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    conditioning_pixel_values = torch.stack([example["conditioning_pixel_values"] for example in examples])
    conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()

    prompt_ids = torch.stack([torch.tensor(example["prompt_embeds"]) for example in examples])

    add_text_embeds = torch.stack([torch.tensor(example["text_embeds"]) for example in examples])
    add_time_ids = torch.stack([torch.tensor(example["time_ids"]) for example in examples])

    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
        "prompt_ids": prompt_ids,
        "unet_added_conditions": {"text_embeds": add_text_embeds, "time_ids": add_time_ids},
    }


def patch_accelerator_for_fp16_training(accelerator):
    org_unscale_grads = accelerator.scaler._unscale_grads_

    def _unscale_grads_replacer(optimizer, inv_scale, found_inf, allow_fp16):
        return org_unscale_grads(optimizer, inv_scale, found_inf, True)

    accelerator.scaler._unscale_grads_ = _unscale_grads_replacer


# 텐서를 이미지로 변환하고 저장하는 함수 추가
def save_tensor_as_image(tensor, save_path):
    """
    PyTorch 텐서를 PIL 이미지로 변환하여 저장합니다.
    
    Args:
        tensor (torch.Tensor): 저장할 이미지 텐서 (C,H,W) 형태
        save_path (str): 이미지를 저장할 경로
    """
    # 텐서가 GPU에 있다면 CPU로 이동
    if tensor.device.type != 'cpu':
        tensor = tensor.cpu()
    
    # 배치 차원이 있다면 제거
    if len(tensor.shape) == 4:
        tensor = tensor[0]
    
    # 정규화 해제 ([-1, 1] 범위를 [0, 1] 범위로 변환)
    tensor = ((tensor + 1) / 2).clamp(0, 1)
    
    # 채널 순서 변경 (C,H,W) -> (H,W,C)
    tensor = tensor.permute(1, 2, 0)
    
    # NumPy 배열로 변환
    array = tensor.numpy()
    
    # RGB 범위를 0-255로 변환
    array = (array * 255).astype(np.uint8)
    
    # PIL 이미지로 변환
    image = Image.fromarray(array)
    
    # 이미지 저장
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    image.save(save_path)
    return image


def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load the tokenizers
    tokenizer_one = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
        use_fast=False,
    )

    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder_one = text_encoder_cls_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )
    vae_path = (
        args.pretrained_model_name_or_path
        if args.pretrained_vae_model_name_or_path is None
        else args.pretrained_vae_model_name_or_path
    )
    vae = AutoencoderKL.from_pretrained(
        vae_path,
        subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
        revision=args.revision if args.pretrained_vae_model_name_or_path is None else None,
        variant=args.variant if args.pretrained_vae_model_name_or_path is None else None,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant, use_safetensors=args.use_safetensors,
    )

    # Image Adapter 및 특성 추출기 로드
    logger.info("Loading Image Adapter, MGN, and CLIP models...")
    
    # MGN 모델 로드
    extractor = MGN().to(accelerator.device)
    if hasattr(args, 'extractor_weight') and args.extractor_weight:
        extractor.load_state_dict(torch.load(args.extractor_weight))
    
    # CLIP 모델 로드
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Image Adapter 로드
    image_adapter = CLIP_Attention_Image_adapter_SDXL(
        extractor_dim=2048, 
        hidden_dim=1024, 
        clip_embeddings_dim=args.clip_embeddings_dim  # args에서 값을 가져옴
    )
    
    # Image Adapter 가중치 로드 (기존 가중치가 있는 경우)
    if hasattr(args, 'image_adapter_weight') and args.image_adapter_weight:
        logger.info(f"Loading existing Image Adapter weights from {args.image_adapter_weight}")
        image_adapter.load_state_dict(torch.load(args.image_adapter_weight))

    # 별도의 ID 이미지 로드 대신 데이터셋의 원본 이미지를 사용할 것이므로 해당 부분 제거
    # 학습 루프에서 배치의 원본 이미지와 캡션을 직접 사용

    if args.load_weights_increaments or args.save_weights_increaments:
        import copy
        orig_unet_sd = copy.deepcopy(unet.state_dict())

    if args.pretrained_unet_model_name_or_path:
        logger.info("Loading existing unet weights")
        unet_sd = load_file(args.pretrained_unet_model_name_or_path)
        if args.load_weights_increaments:
            logger.info("Loading unet weights in increaments")
            for k in unet_sd.keys():
                unet_sd[k] += orig_unet_sd[k]
        unet.load_state_dict(unet_sd)
    else:
        logger.info("Initializing unet weights from scratch")
        pass

    controlnet = ControlNetModel()
    if args.controlnet_model_name_or_path:
        logger.info("Loading existing controlnet weights")
        controlnet.load_state_dict(load_file(args.controlnet_model_name_or_path))
    else:
        logger.info("Initializing controlnet weights from scratch")

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    
    # MGN과 CLIP은 고정 (특성 추출용으로만 사용)
    extractor.requires_grad_(False)
    clip_model.requires_grad_(False)
    
    # Image Adapter는 학습 가능하도록 설정
    image_adapter.requires_grad_(True)

    if args.enable_npu_flash_attention:
        if is_torch_npu_available():
            logger.info("npu flash attention enabled.")
            unet.enable_npu_flash_attention()
        else:
            raise ValueError("npu flash attention requires torch_npu extensions and is supported only on npu devices.")

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        controlnet.enable_gradient_checkpointing()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    if unwrap_model(controlnet).dtype != torch.float32:
        raise ValueError(
            f"Controlnet loaded as datatype {unwrap_model(controlnet).dtype}. {low_precision_error_string}"
        )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )
        args.learning_rate_controlnet = (
            args.learning_rate_controlnet * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.optimizer_type.lower() == "adamw":
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW
        optimizer_kwargs = dict(
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
    elif args.optimizer_type.lower() == "adafactor":
        optimizer_class = transformers.optimization.Adafactor
        optimizer_kwargs = dict(
            relative_step=args.adafactor_relative_step,
            scale_parameter=args.adafactor_scale_parameter,
            warmup_init=args.adafactor_warmup_init,
        )
    else:
        raise ValueError(f"Optimizer type {args.optimizer_type} not supported.")

    # Optimizer creation
    controlnet.train()
    controlnet.requires_grad_(True)
    
    # Image Adapter 학습 설정
    image_adapter.train()
    
    params_to_optimize = [
        {'params': list(controlnet.parameters()), 'lr': args.learning_rate_controlnet},
        {'params': list(image_adapter.parameters()), 'lr': args.learning_rate_controlnet}  # Image Adapter도 controlnet과 같은 학습률 사용
    ]
    
    logger.info(f"Number of trainable parameters in controlnet: {sum(p.numel() for p in controlnet.parameters() if p.requires_grad)}")
    logger.info(f"Number of trainable parameters in image_adapter: {sum(p.numel() for p in image_adapter.parameters() if p.requires_grad)}")

    unet.train()
    unet.requires_grad_(True)
    unet_params = []
    pattern = re.compile(args.unet_trainable_param_pattern)
    for name, param in unet.named_parameters():
        if pattern.match(name):
            param.requires_grad = True
            unet_params.append(param)
        else:
            param.requires_grad = False
    logger.info(f"Number of trainable parameters in unet: {sum(p.numel() for p in unet.parameters() if p.requires_grad)}")
    params_to_optimize.append({'params': unet_params, 'lr': args.learning_rate})
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        **optimizer_kwargs,
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae, unet and text_encoder to device and cast to weight_dtype
    # The VAE is in float32 to avoid NaN losses.
    if args.pretrained_vae_model_name_or_path is not None:
        vae.to(accelerator.device, dtype=weight_dtype)
    else:
        vae.to(accelerator.device, dtype=torch.float32)
    unet.to(accelerator.device, dtype=weight_dtype)
    controlnet = controlnet.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    
    # 추가 모델도 디바이스로 이동
    extractor.to(accelerator.device, dtype=weight_dtype)
    clip_model.to(accelerator.device, dtype=weight_dtype)
    image_adapter.to(accelerator.device, dtype=weight_dtype)

    # FP16 패치 강제 적용
    if accelerator.mixed_precision == "fp16":
        patch_accelerator_for_fp16_training(accelerator)
        # 추가 안정화 코드
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Here, we compute not just the text embeddings but also the additional embeddings
    # needed for the SD XL UNet to operate.
    def compute_embeddings(batch, proportion_empty_prompts, text_encoders, tokenizers, is_train=True):
        original_size = (args.resolution, args.resolution)
        target_size = (args.resolution, args.resolution)
        crops_coords_top_left = (args.crops_coords_top_left_h, args.crops_coords_top_left_w)
        prompt_batch = batch[args.caption_column]

        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            prompt_batch, text_encoders, tokenizers, proportion_empty_prompts, is_train
        )
        add_text_embeds = pooled_prompt_embeds

        # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids])

        prompt_embeds = prompt_embeds.to(accelerator.device)
        add_text_embeds = add_text_embeds.to(accelerator.device)
        add_time_ids = add_time_ids.repeat(len(prompt_batch), 1)
        add_time_ids = add_time_ids.to(accelerator.device, dtype=prompt_embeds.dtype)
        unet_added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

        return {"prompt_embeds": prompt_embeds, **unet_added_cond_kwargs}

    # Let's first compute all the embeddings so that we can free up the text encoders
    # from memory.
    text_encoders = [text_encoder_one, text_encoder_two]
    tokenizers = [tokenizer_one, tokenizer_two]
    train_dataset = get_train_dataset(args, accelerator)
    compute_embeddings_fn = functools.partial(
        compute_embeddings,
        text_encoders=text_encoders,
        tokenizers=tokenizers,
        proportion_empty_prompts=args.proportion_empty_prompts,
    )
    with accelerator.main_process_first():
        from datasets.fingerprint import Hasher

        # fingerprint used by the cache for the other processes to load the result
        # details: https://github.com/huggingface/diffusers/pull/4038#discussion_r1266078401
        new_fingerprint = Hasher.hash(args)
        train_dataset = train_dataset.map(compute_embeddings_fn, batched=True, batch_size=500, new_fingerprint=new_fingerprint)
    # match 느릴 때 batch_size 변경
    del text_encoders, tokenizers
    gc.collect()
    torch.cuda.empty_cache()

    # Then get the training dataset ready to be passed to the dataloader.
    train_dataset = prepare_train_dataset(train_dataset, accelerator)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            args.num_train_epochs * num_update_steps_per_epoch * accelerator.num_processes
        )
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    unet, controlnet, image_adapter, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, controlnet, image_adapter, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps * accelerator.num_processes:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))

        # tensorboard cannot handle list types for config
        tracker_config.pop("validation_prompt")
        tracker_config.pop("validation_image")

        # wandb 설정을 별도로 처리
        if args.report_to == "wandb":
            if is_wandb_available():
                import wandb
                
                # wandb 실행 이름 설정
                if args.tracker_run_name is None:
                    from datetime import datetime
                    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                else:
                    run_name = args.tracker_run_name
                
                # wandb 초기화
                wandb.init(
                    project=args.tracker_project_name,
                    name=run_name,
                    config=tracker_config
                )
            else:
                logger.warning("wandb이 설치되어 있지 않습니다. 'pip install wandb'로 설치하세요.")
        
        # wandb가 아닌 다른 tracker는 기존 방식으로 초기화
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)
    
    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    loss_recorder = LossRecorder(gamma=0.9)

    image_logs = None
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet, controlnet, image_adapter):
                # Convert images to latent space
                if args.pretrained_vae_model_name_or_path is not None:
                    pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
                else:
                    pixel_values = batch["pixel_values"]
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                if args.pretrained_vae_model_name_or_path is None:
                    latents = latents.to(weight_dtype)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # ControlNet conditioning.
                controlnet_image = batch["conditioning_pixel_values"].to(accelerator.device, dtype=controlnet.dtype)
                controls = controlnet(
                    controlnet_image,
                    timesteps,
                )
                controls['scale'] *= args.controlnet_scale_factor
                
                # Image Adapter 적용 (원본 이미지와 캡션을 사용)
                # 원본 이미지(pixel_values)에서 MGN과 CLIP 특성 추출
                
                # MGN 특성 추출 (각 배치 항목에 대해)
                with torch.no_grad():
                    # 이미지 정규화가 이미 되어 있으므로 바로 사용
                    # pixel_values는 [-1, 1] 범위의 정규화된 이미지
                    mgn_outputs = extractor(pixel_values)
                    mgn_features = mgn_outputs[0]  # [batch_size, 2048]
                    
                    # CLIP 이미지 특성 추출
                    # pixel_values를 [0, 1] 범위로 변환
                    norm_pixel_values = (pixel_values * 0.5 + 0.5).clamp(0, 1)
                    
                    # 배치 처리를 위해 리스트 구성
                    batch_images = [img for img in norm_pixel_values.cpu()]
                    
                    # CLIP 처리
                    clip_inputs = clip_processor(images=batch_images, return_tensors="pt")
                    clip_pixel_values = clip_inputs.pixel_values.to(accelerator.device)
                    
                    # CLIP 특성 추출
                    clip_image_features = clip_model.get_image_features(clip_pixel_values)
                
                # Image Adapter를 통한 임베딩 생성
                id_control_embeddings = image_adapter(mgn_features, clip_image_features)
                
                # ID 컨트롤 임베딩 정규화
                id_control_embeddings = F.normalize(id_control_embeddings, p=2, dim=-1)
                
                # ID 컨트롤 임베딩을 unet에 전달하기 위한 처리
                # UNet의 입력 임베딩 차원과 호환되도록 선형 투영 수행
                if id_control_embeddings.shape[-1] != batch["prompt_ids"].shape[-1]:
                    projection_layer = nn.Linear(
                        id_control_embeddings.shape[-1], 
                        batch["prompt_ids"].shape[-1]
                    ).to(device=id_control_embeddings.device, dtype=id_control_embeddings.dtype)
                    nn.init.normal_(projection_layer.weight, std=0.02)
                    nn.init.zeros_(projection_layer.bias)
                    id_control_embeddings = projection_layer(id_control_embeddings)
                
                # 원본 프롬프트 임베딩과 ID 컨트롤 임베딩 블렌딩
                control_blend_weight = args.control_blend_weight if hasattr(args, 'control_blend_weight') else 0.7
                id_control_blend_weight = args.id_control_blend_weight if hasattr(args, 'id_control_blend_weight') else 0.3
                
                # 배치 프롬프트 임베딩 수정 (원본 + ID 컨트롤)
                # Shape 디버깅 정보 출력
                logger.info(f"Debug - prompt_ids shape: {batch['prompt_ids'].shape}, id_control_embeddings shape: {id_control_embeddings.shape}")
                
                # 차원 검사를 더 철저히 수행하여 적절한 블렌딩 방식 결정
                # batch["prompt_ids"] 형태는 보통 [batch_size, seq_len, hidden_dim]
                # id_control_embeddings 형태는 [batch_size, hidden_dim]
                if len(batch["prompt_ids"].shape) == 3 and len(id_control_embeddings.shape) == 2:
                    # id_control_embeddings를 [batch_size, 1, hidden_dim] 형태로 확장
                    expanded_id_embeddings = id_control_embeddings.unsqueeze(1)
                    
                    # 각 시퀀스 위치에 동일한 id_control_embeddings 적용하기 위해 확장
                    # [batch_size, 1, hidden_dim] -> [batch_size, seq_len, hidden_dim]
                    expanded_id_embeddings = expanded_id_embeddings.expand(-1, batch["prompt_ids"].shape[1], -1)
                    
                    # 이제 두 텐서의 shape이 완전히 일치해야 함
                    logger.info(f"After expansion - prompt_ids: {batch['prompt_ids'].shape}, expanded_id_embeddings: {expanded_id_embeddings.shape}")
                    
                    # 블렌딩
                    blended_prompt_embeds = batch["prompt_ids"] * control_blend_weight + expanded_id_embeddings * id_control_blend_weight
                else:
                    # 예상치 못한 shape인 경우, 안전하게 처리
                    logger.warning(f"Unexpected tensor shapes: prompt_ids {batch['prompt_ids'].shape}, id_control_embeddings {id_control_embeddings.shape}")
                    # id_control_embeddings 사용하지 않고 원본 임베딩만 사용
                    blended_prompt_embeds = batch["prompt_ids"]

                # Predict the noise residual
                with accelerator.autocast():
                    model_pred = unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=blended_prompt_embeds,  # 블렌딩된 임베딩 사용
                        added_cond_kwargs=batch["unet_added_conditions"],
                        controls=controls,
                        return_dict=False,
                    )[0]

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = []
                    for p in params_to_optimize:
                        params_to_clip.extend(p["params"])
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # DeepSpeed requires saving weights on every device; saving weights only on the main process would cause issues.
                if accelerator.distributed_type == DistributedType.DEEPSPEED or accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, "checkpoints", f"checkpoint-{global_step}")
                        save_models(
                            accelerator.unwrap_model(unet),
                            accelerator.unwrap_model(controlnet),
                            save_path,
                            args,
                        )
                        
                        # Image Adapter 가중치도 저장
                        adapter_save_path = os.path.join(save_path, "image_adapter.pt")
                        torch.save(accelerator.unwrap_model(image_adapter).state_dict(), adapter_save_path)
                        logger.info(f"Saved state to {save_path} (including Image Adapter)")

            loss = loss.detach().item()
            loss_recorder.add(loss=loss)
            loss_avr: float = loss_recorder.moving_average(window=1000)
            loss_ema: float = loss_recorder.ema
            logs = {"loss/step": loss, 'loss_avr/step': loss_avr, 'loss_ema/step': loss_ema, 'lr/step': lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_path = os.path.join(args.output_dir, "checkpoints", "final")
        save_models(
            accelerator.unwrap_model(unet),
            accelerator.unwrap_model(controlnet),
            save_path,
            args,
            orig_unet_sd if args.save_weights_increaments else None,
        )
        
        # 최종 Image Adapter 가중치 저장
        adapter_save_path = os.path.join(save_path, "image_adapter.pt")
        torch.save(accelerator.unwrap_model(image_adapter).state_dict(), adapter_save_path)
        logger.info(f"Saved final Image Adapter weights to {adapter_save_path}")

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)

# 데이터셋 필터링 및 검증
def filter_dataset(examples):
    """데이터셋에서 문제가 될 수 있는 샘플 필터링"""
    filtered_indices = []
    for i, (img, sketch) in enumerate(zip(examples[args.image_column], examples[args.conditioning_image_column])):
        # 스케치가 너무 희미한지 확인
        sketch_array = np.array(sketch.convert("L"))
        if sketch_array.mean() > 240:  # 대부분이 흰색이면 너무 희미함
            continue
            
        # 이미지와 스케치 사이에 충분한 정보 차이가 있는지
        if calculate_similarity(img, sketch) > 0.9:  # 너무 유사하면 학습 가치가 낮음
            continue
            
        filtered_indices.append(i)
    
    return filtered_indices

# 다양한 CFG 값으로 검증
def log_validation_with_cfg(vae, unet, controlnet, args, accelerator, weight_dtype, step):
    """다양한 CFG 값으로 검증 실행 및 Image Adapter를 통한 추가 컨트롤 적용"""
    logger.info("Loading Image Adapter and MGN models for additional control...")
    
    # MGN 및 CLIP 모델 로드
    extractor = MGN().to(accelerator.device)
    if hasattr(args, 'extractor_weight') and args.extractor_weight:
        extractor.load_state_dict(torch.load(args.extractor_weight))
    
    # CLIP 모델 로드
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(accelerator.device) 
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Image Adapter 로드
    image_adapter = CLIP_Attention_Image_adapter_SDXL(
        extractor_dim=2048, 
        hidden_dim=1024, 
        clip_embeddings_dim=512
    ).to(accelerator.device)
    
    # Image Adapter 가중치 로드 (있는 경우)
    if hasattr(args, 'image_adapter_weight') and args.image_adapter_weight:
        image_adapter.load_state_dict(torch.load(args.image_adapter_weight))
    
    # 결과 저장을 위한 디렉토리 생성
    os.makedirs(f"{args.output_dir}/validation", exist_ok=True)
    os.makedirs(f"{args.output_dir}/validation/id_control", exist_ok=True)
    
    # 검증 이미지와 프롬프트 설정
    if len(args.validation_image) == len(args.validation_prompt):
        validation_images = args.validation_image
        validation_prompts = args.validation_prompt
    elif len(args.validation_image) == 1:
        validation_images = args.validation_image * len(args.validation_prompt)
        validation_prompts = args.validation_prompt
    elif len(args.validation_prompt) == 1:
        validation_images = args.validation_image
        validation_prompts = args.validation_prompt * len(args.validation_image)
    
    # 원본 이미지 설정 (validation_source_image)
    if args.validation_source_image is None:
        # 원본 이미지가 제공되지 않으면 스케치 이미지 자체를 사용
        validation_source_images = validation_images
    else:
        if len(args.validation_source_image) == len(validation_images):
            validation_source_images = args.validation_source_image
        elif len(args.validation_source_image) == 1:
            validation_source_images = args.validation_source_image * len(validation_images)
        else:
            raise ValueError("validation_source_image 개수가 validation_image와 일치하지 않습니다.")
    
    results = []
    cfg_values = [1.0, 3.0, 5.0, 7.0]
    
    # 모든 모델을 평가 모드로 설정
    extractor.eval()
    image_adapter.eval()
    clip_model.eval()
    
    # Blending 가중치 (기존 controlnet과 ID controlnet 간의 가중치)
    control_blend_weight = args.control_blend_weight if hasattr(args, 'control_blend_weight') else 0.7
    id_control_blend_weight = args.id_control_blend_weight if hasattr(args, 'id_control_blend_weight') else 0.3
    
    for cfg in cfg_values:
        for idx, (validation_prompt, validation_image, validation_source_image) in enumerate(
            zip(validation_prompts[:1], validation_images[:1], validation_source_images[:1])
        ):
            # 스케치 이미지 로드
            sketch_image = Image.open(validation_image).convert("RGB")
            sketch_image = sketch_image.resize((args.resolution, args.resolution))
            
            # 원본 이미지 로드 (Image Adapter용)
            source_image = Image.open(validation_source_image).convert("RGB")
            source_image = source_image.resize((args.resolution, args.resolution))
            
            # 원본 이미지 저장 (확인용)
            source_image.save(f"{args.output_dir}/validation/id_control/source_image_step_{step}.png")
            
            # 원본 이미지 특성 추출을 위한 전처리
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
            source_tensor = transform(source_image).unsqueeze(0).to(accelerator.device)
            
            # 원본 이미지에서 특성 추출
            with torch.no_grad():
                # MGN 특성 추출
                mgn_outputs = extractor(source_tensor)
                mgn_features = mgn_outputs[0]  # [batch_size, 2048]
                
                # CLIP 이미지 특성 추출
                # 이미지를 [0, 1] 범위로 변환
                norm_source_tensor = (source_tensor * 0.5 + 0.5).clamp(0, 1)
                
                # CLIP 처리
                clip_inputs = clip_processor(
                    images=[norm_source_tensor[0].cpu()],
                    return_tensors="pt"
                )
                clip_pixel_values = clip_inputs.pixel_values.to(accelerator.device)
                
                # CLIP 특성 추출
                clip_image_features = clip_model.get_image_features(clip_pixel_values)
                
                # Image Adapter를 통한 임베딩 생성
                id_control_embedding = image_adapter(mgn_features, clip_image_features)
                
                # ID 컨트롤 임베딩 정규화
                id_control_embedding = F.normalize(id_control_embedding, p=2, dim=-1)
            
            # 기존 파이프라인에 ID 컨트롤을 적용하기 위한 커스텀 파이프라인 생성
            pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                controlnet=accelerator.unwrap_model(controlnet),
                vae=vae,
                unet=unet,
                torch_dtype=weight_dtype,
                variant=args.variant,
            ).to(accelerator.device)
            
            # 기존 controlnet만 사용하는 이미지 생성 (비교용)
            base_image = pipeline(
                prompt=validation_prompt,
                controlnet_image=sketch_image,
                guidance_scale=cfg,
                controlnet_scale_factor=args.controlnet_scale_factor,
                num_inference_steps=20,
            ).images[0]
            
            # ID 컨트롤 적용 이미지 생성을 위한 파이프라인 설정
            # 파이프라인의 _encode_prompt 메소드를 복사
            original_encode_prompt = pipeline._encode_prompt
            
            # _encode_prompt 메소드 오버라이드
            def custom_encode_prompt(*args, **kwargs):
                prompt_embeds, pooled_prompt_embeds, negative_prompt_embeds, negative_pooled_prompt_embeds = original_encode_prompt(*args, **kwargs)
                
                # ID 컨트롤 임베딩과 프롬프트 임베딩 차원 맞추기 
                if prompt_embeds.shape[-1] != id_control_embedding.shape[-1]:
                    adapter_projection = nn.Linear(
                        id_control_embedding.shape[-1], 
                        prompt_embeds.shape[-1]
                    ).to(device=id_control_embedding.device, dtype=id_control_embedding.dtype)
                    nn.init.normal_(adapter_projection.weight, std=0.02)
                    nn.init.zeros_(adapter_projection.bias)
                    id_control_embedding_projected = adapter_projection(id_control_embedding)
                else:
                    id_control_embedding_projected = id_control_embedding
                
                # 블렌딩 대신 시퀀스 차원에서 결합
                combined_prompt_embeds = torch.cat([prompt_embeds, id_control_embedding_projected], dim=1)
                
                # 네거티브 프롬프트도 동일하게 처리 
                zeros = torch.zeros_like(id_control_embedding_projected)
                combined_negative_prompt_embeds = torch.cat([negative_prompt_embeds, zeros], dim=1)
                
                return combined_prompt_embeds, pooled_prompt_embeds, combined_negative_prompt_embeds, negative_pooled_prompt_embeds
            
            # 파이프라인에 커스텀 메소드 적용
            pipeline._encode_prompt = custom_encode_prompt
            
            # ID 컨트롤 적용 이미지 생성
            blended_image = pipeline(
                prompt=validation_prompt,  # 같은 프롬프트 사용
                controlnet_image=sketch_image,
                guidance_scale=cfg,
                controlnet_scale_factor=args.controlnet_scale_factor,
                num_inference_steps=20,
            ).images[0]
            
            # 결과 저장
            results.append({
                "cfg": cfg, 
                "image": base_image, 
                "prompt": validation_prompt,
                "type": "base"
            })
            
            results.append({
                "cfg": cfg, 
                "image": blended_image, 
                "prompt": f"{validation_prompt} (with image features)",
                "type": "id_blended"
            })
            
            # 개별 이미지 저장
            base_image.save(f"{args.output_dir}/validation/id_control/base_cfg{cfg}_step_{step}.png")
            blended_image.save(f"{args.output_dir}/validation/id_control/blended_cfg{cfg}_step_{step}.png")
    
    # 이미지 그리드로 저장
    base_results = [r for r in results if r["type"] == "base"]
    blended_results = [r for r in results if r["type"] == "id_blended"]
    
    base_grid = make_validation_grid(base_results)
    blended_grid = make_validation_grid(blended_results)
    
    base_grid.save(f"{args.output_dir}/validation/base_cfg_test_step_{step}.png")
    blended_grid.save(f"{args.output_dir}/validation/id_blended_cfg_test_step_{step}.png")
    
    # 모든 결과를 하나의 그리드로 결합
    all_grid = make_validation_grid(results)
    all_grid.save(f"{args.output_dir}/validation/all_cfg_test_step_{step}.png")
    
    logger.info(f"Validation with Image Adapter features completed for step {step}")