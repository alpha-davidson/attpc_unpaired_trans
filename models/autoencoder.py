import torch
from torch.nn import Module

from .encoders import *
from .diffusion import *


class AutoEncoder(Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = PointNetEncoder(zdim=args.latent_dim)
        self.diffusion = DiffusionPoint(
            net = PointwiseNet(point_dim=4, context_dim=args.latent_dim, residual=args.residual), #changed from 3 to 4
            var_sched = VarianceSchedule(
                num_steps=args.num_steps,
                beta_1=args.beta_1,
                beta_T=args.beta_T,
                mode=args.sched_mode
            )
        )

    def encode(self, x):
        """
        Args:
            x:  Point clouds to be encoded, (B, N, d).
        """
        # print("ae encode x shape: ", x.shape)
        code, _ = self.encoder(x)
        return code

    def decode(self, code, num_points, flexibility=0.0, ret_traj=False):
        return self.diffusion.sample(num_points, code, flexibility=flexibility, ret_traj=ret_traj)

    def get_loss(self, x):
        code = self.encode(x)
        loss = self.diffusion.get_loss(x, code)
        return loss


# to do: when changing input dem to four, just initialize model and print shape of output. use debugger 
# input 4D and output the output shape -> if we can find the 4D -> 3D print shapes as we go to see where the shape gets lost