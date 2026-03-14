from __future__ import annotations

from ..core.activations import register_cache_pair_activation
from .blocks import BlockSpec, ResolvedBlock
from .cache_provider import (
    CacheSpec,
    ContractError,
    declares_blocks,
    forward_with_standard_cache,
    resolve_declared_blocks,
)
from .cache_wrapper import ModelCacheWrapper
from .builtins.actor_critic import ActorCriticDiscrete
from .builtins.bert import HFSequenceClassifier
from .builtins.convnet import ConvNetClassifier
from .builtins.mlp import MLP, MLPClassifier, MLPStack
from .builtins.resnet import ResNet, build_resnet18, build_resnet34, build_resnet50
from .registry import ModelContext, build_model, get_model_names, register_model

__all__ = [
    "ActorCriticDiscrete",
    "BlockSpec",
    "CacheSpec",
    "ConvNetClassifier",
    "ContractError",
    "HFSequenceClassifier",
    "MLP",
    "MLPClassifier",
    "MLPStack",
    "ModelCacheWrapper",
    "ModelContext",
    "ResolvedBlock",
    "forward_with_standard_cache",
    "ResNet",
    "build_model",
    "build_resnet18",
    "build_resnet34",
    "build_resnet50",
    "declares_blocks",
    "get_model_names",
    "register_cache_pair_activation",
    "register_model",
    "resolve_declared_blocks",
]
