# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Copyright (c) 2026 Galaxea

"""PolicyInferencer — reusable batched inference for G05 policies."""

from __future__ import annotations

import dataclasses
import logging
import threading
import time
from typing import Any

import torch

from g05.data_processor.processor.mixture_processor import MixtureProcessor
from g05.utils.data.data_utils import collate_fn_pad_sequences, custom_collate_fn
from g05.utils.common.pytorch_utils import dict_apply

logger = logging.getLogger(__name__)


def _sync_if_cuda_available() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@dataclasses.dataclass(slots=True)
class PreparedSample:
    """Preprocessed sample plus its sub_processor reference."""

    obs_dict: dict[str, Any]
    sample: dict[str, Any]
    sub_processor: Any


def resolve_processor(processor, data):
    """Return the matching sub-processor from MixtureProcessor, or single directly.

    Supports both raw_obs (key="embodiment_type") and obs_dict (key="embodiment") forms.
    """
    if isinstance(processor, MixtureProcessor):
        emb = data.get("embodiment_type") or data.get("embodiment")
        if emb is None and len(processor.processors) == 1:
            emb = next(iter(processor.processors))
        return processor[emb]
    return processor


class PolicyInferencer:
    """Unified inference wrapper: obs_dicts in → action dicts out."""

    def __init__(self, policy, processor, device: str = "cuda"):
        self.policy = policy
        self.processor = processor
        self.device = device
        # WebSocket connections dispatch inference via asyncio.to_thread.
        # The shared processor/model (including CUDA graphs and mutable caches)
        # is not reentrant; serialize the complete inference transaction.
        self._inference_lock = threading.Lock()

    # ──────── Public API ────────

    def infer(self, obs_dicts: list[dict]) -> list[dict]:
        """Full pipeline: raw obs_dicts → postprocessed action dicts."""
        results, _ = self.infer_with_timing(obs_dicts)
        return results

    def infer_with_timing(self, obs_dicts: list[dict]) -> tuple[list[dict], dict[str, float]]:
        """Same as :meth:`infer` plus per-stage timing (ms).

        Returns ``(results, {"preprocess_ms", "infer_ms", "postprocess_ms"})``.
        """
        with self._inference_lock:
            return self._infer_with_timing_locked(obs_dicts)

    def _infer_with_timing_locked(self, obs_dicts: list[dict]) -> tuple[list[dict], dict[str, float]]:
        """Caller holds _inference_lock until postprocessing is complete."""
        if not obs_dicts:
            return [], {"preprocess_ms": 0.0, "infer_ms": 0.0, "postprocess_ms": 0.0}

        _sync_if_cuda_available()
        t0 = time.monotonic()
        prepared = [self._prepare(obs_dict) for obs_dict in obs_dicts]
        padding_input_id = prepared[0].sub_processor.pad_token_id
        batch = self._collate([p.sample for p in prepared], padding_input_id=padding_input_id)
        _sync_if_cuda_available()
        t1 = time.monotonic()
        batch = self.forward_batch(batch)
        _sync_if_cuda_available()
        t2 = time.monotonic()
        model_timing = batch.pop("_timing", {})
        cot_texts = batch.get("cot_text")
        results = []
        for i, p in enumerate(prepared):
            action = self._postprocess_single(batch, index=i, sub_processor=p.sub_processor)
            if cot_texts is not None:
                action["_cot_text"] = cot_texts[i]
            results.append(action)
        _sync_if_cuda_available()
        t3 = time.monotonic()
        timing = {
            "preprocess_ms": (t1 - t0) * 1000.0,
            "infer_ms": (t2 - t1) * 1000.0,
            "postprocess_ms": (t3 - t2) * 1000.0,
        }
        if isinstance(model_timing, dict):
            timing.update(model_timing)
        return results, timing

    def infer_one(self, obs_dict: dict) -> dict:
        """Single obs convenience."""
        return self.infer([obs_dict])[0]

    def infer_rtc(self, obs_dict: dict, absolute_prefix: dict, delay_steps: int,
                  soft_steps: int = 4, guidance_weight: float = 1.0):
        """Modified by doulei741, 2026-10-27: full FM-only RTC inference transaction.

        Absolute raw actions are transformed using the new observation state and
        the exact trained normalization/grouping. Gradients are enabled only in
        the action sampler, with respect to its current latent, never accumulated
        into model parameters. Existing unconditioned inference stays unchanged.
        """
        from .helpers.rtc import RTCContext, normalize_absolute_prefix, use
        if not self.policy.continuous_action or self.policy.discrete_action or self.policy.predict_cot:
            raise ValueError('RTC requires FM-only inference without AR/CoT generation')
        with self._inference_lock:
            processor = resolve_processor(self.processor, obs_dict)
            prefix = normalize_absolute_prefix(processor, obs_dict, absolute_prefix).to(self.device)
            context = RTCContext(prefix, int(delay_steps), int(soft_steps), float(guidance_weight))
            with use(context):
                results, timing = self._infer_with_timing_locked([obs_dict])
            # Verify the entire processor roundtrip, including relative joints,
            # grouping/padding and gripper semantics, before exposing a chunk.
            max_error = 0.0
            for key, expected in absolute_prefix.items():
                actual = results[0][key][:, :delay_steps].float().cpu()
                reference = torch.as_tensor(expected, dtype=torch.float32)[:delay_steps].unsqueeze(0)
                if delay_steps:
                    max_error = max(max_error, float((actual-reference).abs().max()))
                    if not torch.allclose(actual, reference, rtol=1e-4, atol=2e-4):
                        raise RuntimeError('RTC committed prefix failed physical-coordinate roundtrip: '+key)
            timing.update(rtc=True, rtc_prefix_steps=prefix.shape[1], rtc_delay_steps=delay_steps)
            timing['rtc_hard_prefix_max_error'] = max_error
            return results[0], timing

    def forward_batch(self, batch: dict) -> dict:
        """Core: to device → predict_action → to cpu."""
        batch = dict_apply(batch, lambda x: x.to(self.device) if isinstance(x, torch.Tensor) else x)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            batch = self.policy.predict_action(batch)
        batch = dict_apply(batch, lambda x: x.cpu() if isinstance(x, torch.Tensor) else x)
        return batch

    # ──────── Private helpers ────────

    def _prepare(self, obs_dict: dict) -> PreparedSample:
        """Preprocess one request and annotate its batching key."""
        sub_processor = resolve_processor(self.processor, obs_dict)
        sample = sub_processor.preprocess(obs_dict)

        sample.pop("action", None)
        sample.pop("action_is_pad", None)
        sample.pop("gt_action", None)

        return PreparedSample(
            obs_dict=obs_dict,
            sample=sample,
            sub_processor=sub_processor,
        )

    @staticmethod
    def _collate(samples: list[dict], padding_input_id: int) -> dict:
        """Collate inference samples, padding text fields when present."""
        batch = [dict(sample) for sample in samples]
        first = batch[0]
        has_text_tokens = {"input_ids", "labels", "attention_mask"} <= set(first.keys())
        if has_text_tokens:
            return collate_fn_pad_sequences(batch, padding_input_id=padding_input_id)

        collated = custom_collate_fn(batch)
        if "samples" in first:
            collated["samples"] = [sample["samples"] for sample in batch]
        sample_meta_keys = ("idx", "task", "embodiment", "dataset_locator", "frequency")
        collated["sample_meta"] = [
            {key: sample.get(key) for key in sample_meta_keys if key in sample} for sample in batch
        ]
        return collated

    @staticmethod
    def _postprocess_single(batch: dict, index: int, sub_processor) -> dict:
        """Slice one sample from a batched model output and postprocess it."""
        item_batch = {
            "action": batch["action"][index : index + 1],
            "proprio": batch["proprio"][index : index + 1],
        }
        if "action_dim_is_pad" in batch:
            item_batch["action_dim_is_pad"] = batch["action_dim_is_pad"][index : index + 1]
        if "proprio_dim_is_pad" in batch:
            item_batch["proprio_dim_is_pad"] = batch["proprio_dim_is_pad"][index : index + 1]
        if "action_op_mask" in batch:
            item_batch["action_op_mask"] = batch["action_op_mask"][index : index + 1]
        action = sub_processor.postprocess(item_batch)["action"]
        # AR omissions describe only the discrete prediction, never the FM
        # tensor selected as the final action when both heads are enabled.
        ar_absent_keys = (
            batch.get("ar_absent_keys") if batch.get("action_source", "ar") == "ar" else None
        )
        absent = (
            ar_absent_keys[index]
            if ar_absent_keys is not None and index < len(ar_absent_keys)
            else set()
        )
        if absent:
            # AR uses grouped names (left_control); postprocess restores raw
            # embodiment keys (left_arm / left_ee_pose). Map before filtering.
            merger = getattr(sub_processor, "action_state_merger", None)
            merge_spec = getattr(merger, "merge_spec", None) or {}
            raw_absent = {
                raw_key
                for key in absent
                for raw_key in merge_spec.get(key, [key])
                if raw_key in action
            }
            if raw_absent:
                action["_absent_keys"] = raw_absent
        return action
