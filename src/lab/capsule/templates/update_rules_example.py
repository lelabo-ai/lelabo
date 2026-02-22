"""Minimal update-rule template for this capsule."""

# from lab.update_rules.registry import UpdateRuleContext, register_update_rule
#
# @register_update_rule("my_rule")
# def build_my_rule(ctx: UpdateRuleContext):
#     # ctx.optimizer is already built by LeLabo.
#     # Return an object compatible with your training flow.
#     # Example:
#     # from lab.update_rules.backprop import Backprop
#     # return Backprop(optimizer=ctx.optimizer, grad_clip=None)
#     raise NotImplementedError
