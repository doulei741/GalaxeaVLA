"""WebSocket policy server with training-inference-consistent action prediction.

Reuses the model/processor loading path from eval_open_loop.py to keep training
and test behavior consistent. Clients send only raw observations via msgpack;
the server returns a denormalized action dict.

Usage:
    # Single embodiment
    python scripts/serve_policy.py \
        --ckpt_path /path/to/checkpoints/step_10000/model_state_dict.pt \
        --host 0.0.0.0 --port 8765 \
        eval_embodiment=galaxea_r1lite

    # Mixture: client routes by the embodiment_type field.
    python scripts/serve_policy.py \
        --ckpt_path /path/to/checkpoints/step_10000/model_state_dict.pt \
        --host 0.0.0.0 --port 8765
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import logging
import time

from g05.utils.websocket import packb, unpackb
import numpy as np
import torch

from pathlib import Path
from typing import Any

import rootutils

rootutils.setup_root(__file__, indicator=".python-version", pythonpath=True)

from g05.utils.config.config_resolvers import register_default_resolvers

register_default_resolvers()

from g05.utils.checkpoint.checkpoint_utils import load_model_from_checkpoint
from g05.utils.data.normalizer import load_dataset_stats_from_json
from g05.utils.common.pytorch_utils import dict_apply
from g05.data_processor.processor.mixture_processor import MixtureProcessor

from g05.utils.data.processor_utils import build_processors
from g05.utils.eval.eval_utils import filter_embodiment
from g05.utils.checkpoint.ckpt_utils import find_run_dir, load_config_from_run_dir
from g05.models.g05.inferencer import (
    PolicyInferencer,
    resolve_processor,
)

import websockets

logger = logging.getLogger(__name__)


def _resolve_stats_path(cfg) -> Path:
    """Resolve dataset stats path for both `last.pt` and `checkpoints/step_x.pt`."""
    run_dir = cfg.get("run_dir", None)
    if run_dir:
        candidate = Path(str(run_dir)) / "dataset_stats.json"
        if candidate.exists():
            return candidate

    ckpt_path = Path(cfg.ckpt_path)
    candidates = [
        ckpt_path.parent / "dataset_stats.json",
        ckpt_path.parent.parent / "dataset_stats.json",
    ]
    if cfg.get("datastatics_path", None):
        candidates.append(Path(str(cfg.datastatics_path)))

    checked = []
    for candidate in candidates:
        checked.append(str(candidate))
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"dataset_stats.json not found (tried: {checked})")


# ──────── setup: model + processor initialization ────────


def setup(cfg, device: str = "cuda"):
    """Load policy and processor, aligned with eval_open_loop.py:362-413."""
    # --- Model loading: meta-device acceleration, skip HF safetensors I/O,
    #     and replay deferred ActionCodecV2 loads. ---
    model = load_model_from_checkpoint(
        cfg.model.model_arch,
        cfg.ckpt_path,
        device=device,
        extra_prefixes=["normalizer."],
        load_config={"preserve_checkpoint_dtype": True},
        use_meta_device=True,  # Load checkpoint weights without a second full FP32 copy.
        eval_mode=False,  # Defer until after bf16 conversion and fp32_params.
    )

    if cfg.model.get("model_weights_to_bf16", True):
        model = model.to(torch.bfloat16)

    model.apply_fp32_params()

    policy = model.eval()
    if hasattr(policy, "action_tokenizer"):
        policy.action_tokenizer.to(device)

    # --- torch.compile, aligned with finetune.py:359-361. ---
    if cfg.model.get("use_torch_compile", False):
        compile_mode = cfg.model.get("torch_compile_mode", "reduce-overhead")
        logger.info(
            "Compiling model with torch.compile (mode=%s, first inference will be slow)...",
            compile_mode,
        )
        policy = torch.compile(policy, mode=compile_mode)

    # --- Normalizer, aligned with train/eval dataset_stats resolution order. ---
    stats_path = _resolve_stats_path(cfg)
    dataset_stats = load_dataset_stats_from_json(stats_path)

    # --- Processor; build_processors handles both single and mixture cases. ---
    processor = build_processors(cfg)
    processor.set_normalizer_from_stats(dataset_stats)
    processor.eval()

    # --- Disable action_filter idle detection during inference. ---
    # DroidIdleActionFilter.forward detects idle frames from action deltas, but
    # inference uses all-zero dummy actions. That would make action_op_mask all
    # False and cause postprocess.backward to zero out the entire prediction.
    # Replace it with BaseActionFilter: forward emits an all-True mask and
    # backward is a no-op.
    from g05.data_processor.transforms.action_filter import BaseActionFilter

    def _neutralize_action_filter(p):
        """Replace the processor action_filter with a no-op BaseActionFilter."""
        safe_filter = BaseActionFilter()
        safe_filter.set_shape_meta(p.shape_meta)
        p.action_filter = safe_filter

    if isinstance(processor, MixtureProcessor):
        for emb_name in processor.processors:
            _neutralize_action_filter(processor.processors[emb_name])
    else:
        _neutralize_action_filter(processor)

    action_horizon = int(cfg.data.action_size)
    if isinstance(processor, MixtureProcessor):
        for emb_name, sub_processor in processor.processors.items():
            sub_processor.action_horizon = action_horizon
    else:
        processor.action_horizon = action_horizon

    return policy, processor


# ──────── input validation ────────


def _parse_task_and_plan(raw_obs: dict) -> None:
    """Parse a [PLAN] marker from task and split it into task + plan.

    txt file format:
        fold the towel
        [PLAN]
        1. Pick up the towel
        2. Fold in half

    raw_obs is unchanged when [PLAN] is absent, preserving backward compatibility.
    """
    task_str = raw_obs.get("task", "")
    if "[PLAN]" in task_str:
        parts = task_str.split("[PLAN]", 1)
        raw_obs["task"] = parts[0].strip()
        raw_obs["plan"] = parts[1].strip()


def _validate_obs(raw_obs: dict, processor) -> None:
    """Validate that client raw_obs satisfies the raw_shape protocol.

    Clients/environments only need to know the raw environment schema.
    shape_meta["shape"] is used only for server-internal transformed
    representations and must not leak into the protocol layer.
    """
    p = resolve_processor(processor, raw_obs)
    sm = p.shape_meta

    # --- Top-level keys. ---
    for required in ("images", "state", "task"):
        if required not in raw_obs:
            raise ValueError(f"raw_obs is missing required field '{required}'")

    if (
        isinstance(processor, MixtureProcessor)
        and len(processor.processors) > 1
        and "embodiment_type" not in raw_obs
    ):
        raise ValueError("Mixture mode requires raw_obs to contain an 'embodiment_type' field")

    if not isinstance(raw_obs["task"], str):
        raise ValueError(f"raw_obs['task'] must be str, got {type(raw_obs['task'])}")

    if "frequency" in raw_obs and not isinstance(raw_obs["frequency"], (int, float, np.number)):
        raise ValueError(
            f"raw_obs['frequency'] must be numeric, got {type(raw_obs['frequency'])}"
        )

    if "coarse_task" in raw_obs and not isinstance(raw_obs["coarse_task"], str):
        raise ValueError(
            f"raw_obs['coarse_task'] must be str, got {type(raw_obs['coarse_task'])}"
        )

    if "plan" in raw_obs and not isinstance(raw_obs["plan"], str):
        raise ValueError(f"raw_obs['plan'] must be str, got {type(raw_obs['plan'])}")

    # --- Image validation. ---
    expected_img_keys = {m["key"] for m in sm["images"]}
    actual_img_keys = set(raw_obs["images"].keys())
    missing = expected_img_keys - actual_img_keys
    if missing:
        raise ValueError(f"images is missing keys defined by shape_meta: {missing}")
    extra = actual_img_keys - expected_img_keys
    if extra:
        logger.warning(f"images contains keys not defined by shape_meta and they will be ignored: {extra}")

    for m in sm["images"]:
        img = raw_obs["images"][m["key"]]
        if not isinstance(img, np.ndarray):
            raise ValueError(f"images['{m['key']}'] must be np.ndarray, got {type(img)}")
        if img.ndim != 3:
            raise ValueError(f"images['{m['key']}'] must be [C,H,W] (3D), got shape={img.shape}")
        if img.dtype != np.uint8:
            logger.warning(f"images['{m['key']}'] dtype={img.dtype}, expected uint8")

    # --- State validation. ---
    expected_state_keys = {m["key"] for m in sm["state"]}
    actual_state_keys = set(raw_obs["state"].keys())
    missing = expected_state_keys - actual_state_keys
    if missing:
        raise ValueError(f"state is missing keys defined by shape_meta: {missing}")
    extra = actual_state_keys - expected_state_keys
    if extra:
        logger.warning(f"state contains keys not defined by shape_meta and they will be ignored: {extra}")

    for m in sm["state"]:
        s = raw_obs["state"][m["key"]]
        if not isinstance(s, np.ndarray):
            raise ValueError(f"state['{m['key']}'] must be np.ndarray, got {type(s)}")
        if s.ndim != 1:
            raise ValueError(f"state['{m['key']}'] must be [D] (1D), got shape={s.shape}")
        expected_dim = m["raw_shape"]
        if s.shape[0] != expected_dim:
            raise ValueError(
                f"state['{m['key']}'] dim={s.shape[0]}, expected raw_shape {expected_dim}"
            )


# ──────── obs -> preprocess input construction ────────


def validated_history(raw_obs, processor, count):
    """Opt-in one-second history for the audited six-observation checkpoint.

    Existing single-frame clients retain their old behavior. This does not
    silently change control clients or invent historical sensor values.
    """
    frames = raw_obs.get('observation_history')
    if frames is None:
        return None
    if count != 6 or not isinstance(frames, list) or len(frames) != count:
        raise ValueError('History diagnostic requires exactly six observations')
    offsets = np.asarray(raw_obs.get('history_offsets_s', []), dtype=float)
    if offsets.shape != (count,) or not np.isfinite(offsets).all():
        raise ValueError('Invalid history timestamps')
    if not np.allclose(offsets, np.arange(-5, 1), atol=.15, rtol=0):
        raise ValueError('Expected one-second history spacing, oldest first')
    for frame in frames:
        _validate_obs(dict(raw_obs, images=frame['images'], state=frame['state']), processor)
        for group in ('images', 'state'):
            for key, value in frame[group].items():
                if not np.isfinite(value).all():
                    raise ValueError('Non-finite history: ' + key)
                if value.shape != raw_obs[group][key].shape:
                    raise ValueError('History shape mismatch: ' + key)
    for group in ('images', 'state'):
        for key, value in raw_obs[group].items():
            if not np.array_equal(frames[-1][group][key], value):
                raise ValueError('History last frame must equal current observation')
    return frames


def build_obs_dict(raw_obs: dict, processor) -> dict:
    """Build the processor.preprocess input from client raw observations.

    Data format is aligned with base_lerobot_dataset.py:304-323 __getitem__ output.
    Client input always uses raw_shape; shape appears only after processor-internal
    transforms. The time dimension is filled with dummy actions for num_obs_steps
    only so the merger can produce action_dim_is_pad.
    """
    raw_obs = checkpoint_observation(raw_obs, processor)
    _validate_obs(raw_obs, processor)

    p = resolve_processor(processor, raw_obs)
    num_obs_steps = p.num_obs_steps
    history = validated_history(raw_obs, processor, num_obs_steps)
    action_horizon = getattr(p, "action_horizon", num_obs_steps)

    # Keep only keys defined in shape_meta, matching dataset.__getitem__ behavior.
    expected_img_keys = {m["key"] for m in p.shape_meta["images"]}
    expected_state_keys = {m["key"] for m in p.shape_meta["state"]}

    data = {
        "images": {
            k: torch.from_numpy(np.array(v, copy=True))
            .unsqueeze(0)
            .expand(num_obs_steps, -1, -1, -1)
            for k, v in raw_obs["images"].items()
            if k in expected_img_keys
        },  # [C,H,W] → [num_obs_steps, C, H, W]
        "state": {
            k: torch.from_numpy(np.array(v, copy=True))
            .unsqueeze(0)
            .expand(num_obs_steps, -1)
            .float()
            for k, v in raw_obs["state"].items()
            if k in expected_state_keys
        },  # [D] → [num_obs_steps, D]
        "task": raw_obs["task"],
        # dummy zeros, only for making merger.forward generate the correct action_dim_is_pad.
        "action": {
            m["key"]: torch.zeros(action_horizon, m["raw_shape"]) for m in p.shape_meta["action"]
        },
        "action_is_pad": torch.ones(action_horizon, dtype=torch.bool),
        "state_is_pad": torch.zeros(num_obs_steps, dtype=torch.bool),
        "image_is_pad": torch.zeros(num_obs_steps, dtype=torch.bool),
        "idx": 0,
    }
    if "coarse_task" in raw_obs:
        data["coarse_task"] = raw_obs["coarse_task"]
    if history is not None:
        data['images'] = {k: torch.from_numpy(np.stack([f['images'][k] for f in history]))
                          for k in expected_img_keys}
        data['state'] = {k: torch.from_numpy(np.stack([f['state'][k] for f in history])).float()
                         for k in expected_state_keys}
    if "plan" in raw_obs:
        data["plan"] = raw_obs["plan"]
    frequency = raw_obs.get("frequency", 15)
    data["frequency"] = float(frequency)
    if isinstance(processor, MixtureProcessor):
        data["embodiment"] = raw_obs.get("embodiment_type") or next(iter(processor.processors))
    return data


# ──────── WebSocket handler ────────


def _serialize_error(code: int, message: str) -> dict[str, Any]:
    """Encode service-side errors in a stable payload."""
    return {"error": {"code": code, "message": message}}


class ChunkedPolicyWrapper:
    """Wraps model inference with chunk caching and step-by-step serving.

    Decides when to call the model (recompute) vs serve from cache.
    One instance per client connection.

    Args:
        inferencer: PolicyInferencer for direct model inference.
        processor: processor for building obs dicts.
        action_steps: steps to serve per inference.
            1 = RTC (recompute every call), >1 = chunk mode.
    """

    def __init__(self, inferencer, processor, *, action_steps: int = 16):
        self.inferencer = inferencer
        self.processor = processor
        self.action_steps = action_steps
        self._cached_chunk = None  # {part: ndarray[chunk_size, dim]}
        self._chunk_step = 0
        self._cot_text = None  # str or None, from last recompute

    @property
    def need_obs(self) -> bool:
        """Whether the next call to get_action needs a real observation."""
        return self._cached_chunk is None or self._chunk_step >= self.action_steps

    async def get_action(self, raw_obs: dict) -> tuple[dict, str | None]:
        """Return single-step action ``{part: ndarray[dim]}``.

        Calls the model when the chunk is exhausted, otherwise serves from cache.
        """
        if self.need_obs:
            if not raw_obs:
                raise ValueError("need obs for recompute but got empty")
            t0 = time.monotonic()
            _parse_task_and_plan(raw_obs)
            obs_dict = build_obs_dict(raw_obs, self.processor)
            actions = await asyncio.to_thread(self.inferencer.infer, [obs_dict])
            action = actions[0]
            self._cot_text = action.pop("_cot_text", None)
            # Only AR-selected outputs carry these raw-key omissions. FM
            # outputs must not be filtered using the separate AR prediction.
            absent_keys = action.pop("_absent_keys", set())
            dropped_keys = set(absent_keys).intersection(action)
            for key in dropped_keys:
                del action[key]
            if dropped_keys:
                logger.warning(
                    "Dropped absent action keys %s (emb=%s); client must handle missing keys.",
                    sorted(dropped_keys),
                    raw_obs.get("embodiment_type", "?"),
                )
            self._cached_chunk = dict_apply(
                action, lambda x: x[0].numpy() if isinstance(x, torch.Tensor) else x
            )
            self._chunk_step = 0
            logger.info(
                "Recompute: %.1fms (next %d from cache)",
                (time.monotonic() - t0) * 1000,
                self.action_steps - 1,
            )

        single_step = {}
        for part, arr in self._cached_chunk.items():
            if arr.ndim >= 1 and self._chunk_step < arr.shape[0]:
                single_step[part] = arr[self._chunk_step]
            else:
                single_step[part] = arr[0] if arr.ndim >= 1 else arr
        self._chunk_step += 1
        return single_step, self._cot_text

    def reset(self):
        """Invalidate cache (e.g. on episode boundary)."""
        self._cached_chunk = None
        self._chunk_step = 0


def summarize_chunk(chunk):
    """Read-only diagnostic of the same inference; never changes serving index."""
    summary = {}
    for key in ('left_arm', 'right_arm', 'left_gripper', 'right_gripper'):
        if key not in chunk:
            continue
        arr = np.asarray(chunk[key])
        if arr.ndim != 2 or not np.isfinite(arr).all():
            raise ValueError('Invalid diagnostic chunk: ' + key)
        summary[key] = {
            'shape': list(arr.shape),
            'steps': {str(i): arr[i].tolist() for i in sorted({0, 8, 16, 31, len(arr)-1}) if 0 <= i < len(arr)},
        }
    return summary


def build_handshake(processor, action_steps):
    """Expose raw wire dimensions, not padded model/grouped dimensions."""
    def shape_meta(p):
        return {"action": [
            {"key": str(m["key"]), "raw_shape": int(m["raw_shape"])}
            for m in p.shape_meta["action"]
        ]}

    metadata = {"action_steps": action_steps}
    if isinstance(processor, MixtureProcessor):
        by_embodiment = {str(k): shape_meta(p) for k, p in processor.processors.items()}
        metadata["shape_meta_by_embodiment"] = by_embodiment
        if len(by_embodiment) == 1:
            metadata["shape_meta"] = next(iter(by_embodiment.values()))
    else:
        metadata["shape_meta"] = shape_meta(processor)
    return metadata


async def handler(ws, inferencer, processor, action_steps=16):
    """Handle a single WebSocket connection.

    Creates a per-connection ``ChunkedPolicyWrapper`` that manages the
    recompute/cache logic.  The handler only does WebSocket I/O.
    """
    client = ws.remote_address
    logger.info(f"Client connected: {client} (action_steps={action_steps})")

    await ws.send(packb(build_handshake(processor, action_steps)))
    policy = ChunkedPolicyWrapper(inferencer, processor, action_steps=action_steps)

    try:
        async for msg in ws:
            t0 = time.monotonic()
            try:
                raw_obs = unpackb(msg)
                if isinstance(raw_obs, dict) and raw_obs.get("__reset__"):
                    policy.reset()
                    await ws.send(packb({"__reset__": True}))
                    continue
                action, cot_text = await policy.get_action(raw_obs)
                resp = {"action": action, "need_obs": policy.need_obs}
                if raw_obs.get('__diagnostic_chunk__') is True:
                    resp['diagnostic_chunk'] = summarize_chunk(policy._cached_chunk)
                if cot_text is not None:
                    resp["cot_text"] = cot_text
                await ws.send(packb(resp))
            except ValueError as exc:
                await ws.send(packb(_serialize_error(400, str(exc))))
            except Exception as exc:
                logger.exception("Inference request failed")
                await ws.send(packb(_serialize_error(500, str(exc))))
            dt = (time.monotonic() - t0) * 1000
            logger.debug(f"{'Infer' if policy._chunk_step == 1 else 'Cache'}: {dt:.1f}ms")
    except websockets.exceptions.ConnectionClosed:
        logger.info(f"Client disconnected: {client}")


async def serve(
    handler_fn,
    policy,
    processor,
    host: str,
    port: int,
    *,
    device: str = "cuda",
    action_steps: int = 16,
):
    """Start the WebSocket server."""
    inferencer = PolicyInferencer(policy, processor, device=device)
    bound_handler = functools.partial(
        handler_fn,
        inferencer=inferencer,
        processor=processor,
        action_steps=action_steps,
    )
    async with websockets.serve(bound_handler, host, port, max_size=None):
        mode = "RTC" if action_steps == 1 else f"chunk({action_steps})"
        logger.info(
            "Policy server listening on ws://%s:%d (mode=%s, device=%s)",
            host,
            port,
            mode,
            device,
        )
        await asyncio.Future()  # run forever


# ──────── main ────────


def checkpoint_embodiment(cfg, requested):
    """Resolve the public A1Z name without relabeling checkpoint statistics."""
    datasets = cfg.data.get("embodiment_datasets", {})
    available = {str(item.embodiment_type) for item in datasets.values()}
    if requested == "galaxea_a1z" and requested not in available:
        # The pinned public base checkpoint stores this legacy key in its stats.
        if "galaxea_r1lite" in available:
            return "galaxea_r1lite"
    return requested


def expose_a1z_processor(processor, requested, selected):
    """Add the public alias only after normalization has loaded under its old key."""
    if requested == "galaxea_a1z" and selected == "galaxea_r1lite":
        if not isinstance(processor, MixtureProcessor):
            raise ValueError("The legacy A1Z base profile requires a mixture processor")
        if "galaxea_a1z" in processor.processors:
            raise ValueError("Refusing to replace an existing A1Z processor")
        processor.processors["galaxea_a1z"] = processor.processors[selected]
        processor._a1z_checkpoint_embodiment = selected


def checkpoint_observation(raw_obs, processor):
    """Keep the base model's internal identity while exposing A1Z on the wire."""
    selected = getattr(processor, "_a1z_checkpoint_embodiment", None)
    if selected and raw_obs.get("embodiment_type") == "galaxea_a1z":
        return dict(raw_obs, embodiment_type=selected)
    return raw_obs


def main():
    parser = argparse.ArgumentParser(
        description="WebSocket policy server (training-inference consistent)"
    )
    parser.add_argument("--ckpt_path", required=True, help="Path to checkpoint file")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", default="cuda", help="CUDA device for model inference")
    parser.add_argument(
        "--action_steps",
        type=int,
        default=16,
        help="Steps to serve per inference. >1=chunk mode (default: 16), 1=RTC",
    )
    args, remaining = parser.parse_known_args()

    # remaining are key=value overrides (same syntax as Hydra overrides)
    overrides = [r for r in remaining if "=" in r]

    # --- Config loading, aligned with eval_open_loop.py:429-438. ---
    run_dir = find_run_dir(args.ckpt_path)
    print(f"Found run dir: {run_dir}")

    cfg = load_config_from_run_dir(run_dir, args.ckpt_path, overrides)

    # Support eval_embodiment=<embodiment_type> filtering to a single embodiment_type.
    eval_embodiment = cfg.get("eval_embodiment", None)
    selected_embodiment = checkpoint_embodiment(cfg, eval_embodiment)
    if selected_embodiment and "embodiment_datasets" in cfg.data:
        filter_embodiment(cfg, selected_embodiment)

    # Use the unified Rich format from overwatch to avoid basicConfig behavior
    # depending on import-time side effects.
    from g05.utils.logging.logging_config import setup_logging

    setup_logging(log_level=logging.INFO, is_main_process=True)

    from g05.utils.logging.banner import print_banner

    print_banner(subtitle="Policy Server")

    policy, processor = setup(cfg, device=args.device)
    expose_a1z_processor(processor, eval_embodiment, selected_embodiment)
    logger.info("Model and processor loaded on %s.", args.device)

    asyncio.run(
        serve(
            handler,
            policy,
            processor,
            args.host,
            args.port,
            device=args.device,
            action_steps=args.action_steps,
        )
    )


if __name__ == "__main__":
    main()
