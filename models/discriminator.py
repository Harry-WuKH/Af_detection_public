# models/domain_discriminator.py
import torch
import torch.nn as nn
from torch.autograd import Function



#================================================================
#======DANN Domain Adversial Neural Network Discriminator =======
#================================================================
class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None

def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)

class DomainDiscriminator(nn.Module):
    """
    Domain classifier for DANN.
    input : feat   (B, in_dim)
    output: logits (B, n_domains)
    """
    def __init__(self, in_dim, hidden_dim=256, n_domains=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, n_domains),
        )
    def forward(self, feat, use_grl=False, lambd=1.0):
        """
        use_grl=False:
            normal classifier forward, used in iterative DANN

        use_grl=True:
            apply GRL before classifier, used in standard joint-update DANN
        """
        if use_grl:
            feat = grad_reverse(feat, lambd)
        return self.net(feat)

