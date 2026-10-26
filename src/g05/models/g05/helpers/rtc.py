# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Modified by doulei741, 2026-10-27: inference-time RTC for G0.5 FM.
"""Soft prefix guidance with a full input-Jacobian VJP, plus committed-prefix projection.

Reference: Black et al., Real-Time Execution of Action Chunking Flow Policies,
arXiv:2506.07339. Prefixes are already re-anchored and normalized by the processor.
The ContextVar is transaction-scoped and reset even on inference failure.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import math
import torch


@dataclass(frozen=True)
class RTCContext:
    prefix: torch.Tensor
    delay_steps: int
    soft_steps: int = 4
    guidance_weight: float = 5.0

    def __post_init__(self):
        if self.prefix.ndim!=3 or not torch.isfinite(self.prefix).all():
            raise ValueError('RTC prefix must be finite [B,T,D]')
        if not 0<=self.delay_steps<=self.prefix.shape[1]:
            raise ValueError('RTC delay exceeds available committed prefix')
        if self.soft_steps<0 or not math.isfinite(self.guidance_weight) or not 0<self.guidance_weight<=10:
            raise ValueError('Invalid RTC soft window or guidance weight')


_active=ContextVar('g05_rtc_context',default=None)


def current():
    return _active.get()


@contextmanager
def use(context):
    token=_active.set(context)
    try:
        yield
    finally:
        _active.reset(token)


def weights(context,horizon,device,dtype):
    if context.prefix.shape[1]>horizon:
        raise ValueError('RTC prefix longer than trained prediction horizon')
    out=torch.zeros(horizon,device=device,dtype=dtype)
    hard=context.delay_steps
    out[:hard]=1
    end=min(len(out),context.prefix.shape[1],hard+context.soft_steps)
    if end>hard:
        # Soft constraints reach zero at the far end of the overlap.
        out[hard:end]=torch.linspace(1,0,end-hard+1,device=device,dtype=dtype)[1:]
    return out.view(1,horizon,1)


def project_prefix(action,initial_noise,t,context,pi_convention):
    n=context.delay_steps
    if not n:
        return action
    noise_time=t if pi_convention else 1-t
    noise_time=noise_time.reshape(-1,1,1).float()
    action=action.clone()
    action[:,:n]=noise_time*initial_noise[:,:n].float()+(1-noise_time)*context.prefix[:,:n].to(action)
    return action


def guided_velocity(action,t,velocity_fn,context,pi_convention,pad_mask=None):
    """Differentiate through the FM velocity, not merely the identity x_t term."""
    with torch.enable_grad():
        x=action.detach().float().requires_grad_(True)
        velocity=velocity_fn(x)
        canonical_v=velocity if pi_convention else -velocity
        noise_time=(t if pi_convention else 1-t).float().reshape(-1,1,1)
        endpoint=x-noise_time*canonical_v.float()
        target=torch.zeros_like(endpoint)
        target[:,:context.prefix.shape[1]]=context.prefix.to(target)
        w=weights(context,x.shape[1],x.device,x.dtype)
        if pad_mask is not None:
            w=w.expand_as(x).masked_fill(pad_mask,0)
        error=((target-endpoint)*w).detach()
        correction=torch.autograd.grad(endpoint,x,grad_outputs=error,retain_graph=False,create_graph=False)[0]
        eps=1e-5
        tau=noise_time.clamp(eps,1-eps)
        scale=((tau.square()+(1-tau).square())/(tau*(1-tau))).clamp(max=context.guidance_weight)
        guided=canonical_v.float()-scale*correction
        result=guided if pi_convention else -guided
    if not torch.isfinite(result).all():
        raise RuntimeError('RTC guidance produced non-finite velocity')
    return result.detach()


def normalize_absolute_prefix(processor,obs_dict,absolute_prefix):
    """Use the *current observation* and original processor for relative/norm/group/pad.

    This deliberately reuses the training pipeline rather than subtracting state or
    concatenating groups by hand. No RGB preprocessing or tokenization is needed.
    """
    from copy import deepcopy
    expected={entry['key']:int(entry['raw_shape']) for entry in processor.shape_meta['action']}
    if set(absolute_prefix)!=set(expected):
        raise ValueError('RTC absolute prefix has incompatible action groups')
    arrays={key:torch.as_tensor(value,dtype=torch.float32) for key,value in absolute_prefix.items()}
    lengths={v.shape[0] for v in arrays.values() if v.ndim==2}
    if len(lengths)!=1:
        raise ValueError('RTC prefix must have equal nonempty time dimensions')
    length=next(iter(lengths));horizon=int(processor.action_horizon)
    if not 1<=length<=horizon:
        raise ValueError('RTC prefix length outside trained horizon')
    data=deepcopy(obs_dict);data['action']={}
    for key,value in arrays.items():
        if value.shape!=(length,expected[key]) or not torch.isfinite(value).all():
            raise ValueError('RTC prefix shape/non-finite mismatch')
        data['action'][key]=torch.cat([value,value[-1:].expand(horizon-length,-1)],dim=0)
    data['action_is_pad']=torch.arange(horizon)>=length
    processed=processor.preprocess_action_state(data)
    return processed['action'][:length].unsqueeze(0)
