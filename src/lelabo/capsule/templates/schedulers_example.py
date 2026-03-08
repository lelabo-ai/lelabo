"""Minimal scheduler template for this capsule."""

# import torch
# from lelabo.schedulers import SchedulerContext, register_scheduler
#
# @register_scheduler("my_cosine")
# def build_my_cosine(ctx: SchedulerContext):
#     params = ctx.scheduler_params()
#     t_max = params.pop("T_max", None)
#     if t_max is None:
#         if ctx.epochs is None:
#             raise ValueError("my_cosine requires T_max or a known epochs horizon.")
#         t_max = int(ctx.epochs)
#     return torch.optim.lr_scheduler.CosineAnnealingLR(
#         ctx.optimizer,
#         T_max=int(t_max),
#         **params,
#     )
