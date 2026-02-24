"""Minimal optimizer template for this capsule."""

# import torch
# from lab.optimizers import OptimizerContext, register_optimizer
#
# @register_optimizer("my_adamw")
# def build_my_adamw(ctx: OptimizerContext):
#     params = ctx.optimizer_params()
#     return torch.optim.AdamW(
#         ctx.params,
#         lr=ctx.lr,
#         weight_decay=ctx.weight_decay,
#         **params,
#     )
