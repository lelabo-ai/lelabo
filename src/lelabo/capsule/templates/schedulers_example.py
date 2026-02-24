"""Minimal scheduler template for this capsule."""

# import torch
# from lelabo.schedulers import SchedulerContext, register_scheduler
#
# @register_scheduler("my_cosine")
# def build_my_cosine(ctx: SchedulerContext):
#     params = ctx.scheduler_params()
#     t_max = int(params.pop("T_max", ctx.epochs or 10))
#     return torch.optim.lr_scheduler.CosineAnnealingLR(
#         ctx.optimizer,
#         T_max=t_max,
#         **params,
#     )
