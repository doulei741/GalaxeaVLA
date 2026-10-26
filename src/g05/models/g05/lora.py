# This file is part of G05_A1Z, licensed under the Galaxea Dynamics A1Z License Agreement.
# See LICENSE.txt in the project root; upstream components retain their applicable licenses.

"""LoRA for G05 Linear projections, retaining upstream base checkpoint key names.

Checkpoints contain base weights plus adapters. They are not PEFT adapter-only
exports. Configured policies recreate the adapters before checkpoint loading.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Linear):
    def __init__(self, base, rank, alpha):
        nn.Module.__init__(self)
        self.in_features, self.out_features = base.in_features, base.out_features
        self.weight, self.bias = base.weight, base.bias
        self.scaling = float(alpha) / rank
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features,
                                             device=base.weight.device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank,
                                             device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        base = F.linear(x, self.weight, self.bias)
        delta = F.linear(F.linear(x.to(self.lora_A.dtype), self.lora_A), self.lora_B)
        return base + delta.to(base.dtype) * self.scaling


def configure_lora(policy, config):
    rank = int(config.get('rank', 8))
    alpha = float(config.get('alpha', 16))
    if rank < 1 or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError('LoRA rank and alpha must be positive')
    targets = set(config.get('targets', ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                                         'gate_proj', 'up_proj', 'down_proj']))
    modules = [(name, module) for name, module in policy.named_modules()
               if name.startswith(('model.vlm.', 'model.action_expert.'))
               and name.rsplit('.', 1)[-1] in targets and type(module) is nn.Linear]
    if not modules:
        raise ValueError('No G05 LoRA target Linear modules matched')
    policy.requires_grad_(False)
    for name, module in modules:
        parent, leaf = name.rsplit('.', 1)
        setattr(policy.get_submodule(parent), leaf, LoRALinear(module, rank, alpha))
    return [name for name, _ in modules]


def validate_lora_checkpoint(model, state_dict, allow_base=False):
    expected = {k for k in model.state_dict() if k.endswith(('.lora_A', '.lora_B'))}
    supplied = {k for k in state_dict if k.endswith(('.lora_A', '.lora_B'))}
    if not expected and not supplied:
        return
    if allow_base and expected and not supplied:
        return
    if expected != supplied:
        raise ValueError('LoRA checkpoint/config mismatch: use its saved .hydra/config.yaml; '
                         'only initial training may initialize adapters from a base checkpoint')
    for key in expected:
        if model.state_dict()[key].shape != state_dict[key].shape:
            raise ValueError('LoRA checkpoint rank mismatch: ' + key)
