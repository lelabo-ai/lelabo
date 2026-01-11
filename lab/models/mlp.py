# lab/models/mlp.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, num_classes: int, activation="heaviside"):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes

        dims = [in_dim] + [hidden_dim] * num_layers + [num_classes]
        self.linears = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(len(dims)-1)])

        if activation == "relu":
            self.activation = "relu"
            
        elif activation == "tanh":
            self.activation = "tanh"
            
        elif activation == "sigmoid":
            self.activation = "sigmoid"
        
        elif activation == 'heaviside':
            self.activation = "heaviside"
            
        else:
            raise ValueError("Only relu is implemented in this minimal scaffold.")

    def act(self, x):
        if self.activation == "relu":
            return F.relu(x)
        elif self.activation == "tanh":
            return torch.tanh(x)
        elif self.activation == "sigmoid":
            return torch.sigmoid(x)
        elif self.activation == "heaviside":
            return (x >= 0).to(x.dtype)
        raise RuntimeError

    def act_deriv_from_preact(self, z):
        if self.activation == "relu":
            return (z > 0).to(z.dtype)
        raise RuntimeError

    def forward(self, x, return_cache: bool = False):
        cache = {"inputs": [], "preacts": [], "acts": []}
        a = x
        if return_cache:
            cache["acts"].append(a)

        # hidden layers
        for i in range(len(self.linears) - 1):
            cache["inputs"].append(a) if return_cache else None
            z = self.linears[i](a)
            cache["preacts"].append(z) if return_cache else None
            a = self.act(z)
            cache["acts"].append(a) if return_cache else None

        # output logits
        cache["inputs"].append(a) if return_cache else None
        logits = self.linears[-1](a)
        cache["preacts"].append(logits) if return_cache else None

        return (logits, cache) if return_cache else logits
