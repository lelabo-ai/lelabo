"""Optimizer templates for this capsule."""

import torch

from lelabo.optimizers import OptimizerContext, register_optimizer


# Uncomment this block for the official capsule optimizer golden path:
#   mnist + cnn + bp + capsule_sgd
# Then run:
#   lelabo list optimizers
#   lelabo train supervised --config configs/train.supervised.capsule_optimizer.toml
#
# @register_optimizer("capsule_sgd")
# def build_capsule_sgd(ctx: OptimizerContext):
#     # Minimal builder using the normalized context.
#     return torch.optim.SGD(
#         ctx.params,
#         lr=ctx.lr,
#         weight_decay=ctx.weight_decay,
#     )
#
#
# @register_optimizer("my_adamw")
# def build_my_adamw(ctx: OptimizerContext):
#     # Advanced example with extra params from optimizer.params.
#     params = ctx.optimizer_params()
#     return torch.optim.AdamW(
#         ctx.params,
#         lr=ctx.lr,
#         betas=tuple(params.get("betas", (0.9, 0.999))),
#         eps=float(params.get("eps", 1e-8)),
#         weight_decay=ctx.weight_decay,
#         amsgrad=bool(params.get("amsgrad", False)),
#     )
