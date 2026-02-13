from .base import *
from .backprop import *
from .feedbackalignment import *
from .dfa import *
from .targetprop import *
from .local_probe_bert import *
from .local_probe_blocks import *
from .local_probe_mlp import *
from .softhebb import *
from .kp import *
from .scl import *
from .kp3 import *
from .dni import *
from .ga_rules import *

from .registry import UpdateRuleContext, build_update_rule, get_update_rule_names, register_update_rule

# optionnel mais recommandé: force l’enregistrement des builders
from . import builders  # noqa: F401
