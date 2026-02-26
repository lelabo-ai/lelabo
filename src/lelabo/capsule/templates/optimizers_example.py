"""Optimizer templates for this capsule."""

# import torch
#
# from lelabo.optimizers import OptimizerContext, register_optimizer
#
#
# @register_optimizer("my_sgd")
# def build_my_sgd(ctx: OptimizerContext):
#     # Minimal builder using the normalized context.
#     return torch.optim.SGD(
#         ctx.params,
#         lr=ctx.lr,
#         momentum=ctx.momentum,
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

