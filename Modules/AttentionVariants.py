
import torch
import torch.nn as nn

# factory design, maybe improve later
class AttentionVariants(nn.Module):
    def __init__(self, attended_dim: int, query_dim: int, att_type: str):
        super(AttentionVariants, self).__init__()
        if att_type == 'general':
            self.att = AttentionGeneral(dim_W=query_dim)
        elif att_type == 'scaled_dot':
            self.att = AttentionScaledDotProduct(dim_scale=attended_dim)
        elif att_type == 'dot':
            self.att = AttentionDotProduct()
        elif att_type == 'concat':
            self.att = AttentionConcat(in_features=attended_dim+query_dim, out_features=query_dim)
        elif att_type == 'MLP':
            self.att = AttentionMLP(in_features=attended_dim, out_features=query_dim)
        else:
            raise ValueError("options for attention are: Biaffine, ScaledDotProduct, DotProduct, MLP")

    def forward(self, a, b):
        return self.att(a, b)


class AttentionGeneral(nn.Module):
    def __init__(self, dim_W: int):
        super(AttentionGeneral, self).__init__()

        self.W = nn.Linear(in_features=dim_W, out_features=dim_W)

    def forward(self, a, b):
        score = torch.sum(a*self.W(b), dim=-1, keepdim=True)
        return score


class AttentionScaledDotProduct(nn.Module):
    def __init__(self, dim_scale: int):
        super(AttentionScaledDotProduct, self).__init__()
        self.dim_scale = dim_scale
        self.dot = AttentionDotProduct()

    def forward(self, a, b):
        score = self.dot(a,b)
        score /= (self.dim_scale ** 0.5)
        return score


class AttentionDotProduct(nn.Module):
    def __init__(self, ):
        super(AttentionDotProduct, self).__init__()

    def forward(self, a, b):
        score = torch.sum(a*b, dim=-1, keepdim=True)
        return score


class AttentionMLP(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super(AttentionMLP, self).__init__()

        self.W1 = nn.Linear(in_features=in_features, out_features=in_features)
        self.W2 = nn.Linear(in_features=out_features, out_features=out_features)
        self.activation = nn.Tanh()
        self.U = nn.Linear(in_features=out_features, out_features=1)

    def forward(self, a, b):
        score = self.U(self.activation(self.W1(a) + self.W2(b)))
        return score

class AttentionSimpleConcat(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super(AttentionSimpleConcat, self).__init__()

        # length of src element representation ; length of trg element representation
        self.W = nn.Linear(in_features=in_features, out_features=out_features)
        self.activation = nn.Tanh()

    def forward(self, a, b):
        to_score = torch.cat([a, b], dim=2)
        score = self.activation(self.W(to_score))
        return score


class AttentionConcat(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super(AttentionConcat, self).__init__()

        # length of src element representation ; length of trg element representation
        self.simple_concat = AttentionSimpleConcat(in_features=in_features, out_features=out_features)
        self.U = nn.Linear(in_features=out_features, out_features=1)

    def forward(self, a, b):
        simple_concat_out = self.simple_concat(a,b)
        score = self.U(simple_concat_out)
        return score



