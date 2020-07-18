


import torch
import torch.nn as nn
import torch.nn.functional as F

class LTM_transfer(nn.Module):
    def __init__(self,md=4, stride=1):
        super(LTM_transfer, self).__init__()
        self.md = md #displacement (default = 4pixels)
        self.range = (md*2 + 1) ** 2 #(default = (4x2+1)**2 = 81)
        self.grid = None
        self.Channelwise_sum = None

        d_u = torch.linspace(-self.md * stride, self.md * stride, 2 * self.md + 1).view(1, -1).repeat((2 * self.md + 1, 1)).view(self.range, 1)  # (25,1)
        d_v = torch.linspace(-self.md * stride, self.md * stride, 2 * self.md + 1).view(-1, 1).repeat((1, 2 * self.md + 1)).view(self.range, 1)  # (25,1)
        self.d = torch.cat((d_u, d_v), dim=1).cuda()  # (25,2)

    def L2normalize(self, x, d=1):
        eps = 1e-6
        norm = x ** 2
        norm = norm.sum(dim=d, keepdim=True) + eps
        norm = norm ** (0.5)
        return (x/norm)

    def UniformGrid(self, Input):
        '''
        Make uniform grid
        :param Input: tensor(N,C,H,W)
        :return grid: (1,2,H,W)
        '''
        # torchHorizontal = torch.linspace(-1.0, 1.0, W).view(1, 1, 1, W).expand(N, 1, H, W)
        # torchVertical = torch.linspace(-1.0, 1.0, H).view(1, 1, H, 1).expand(N, 1, H, W)
        # grid = torch.cat([torchHorizontal, torchVertical], 1).cuda()

        _, _, H, W = Input.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, 1, 1, W).expand(1, 1, H, W)
        yy = torch.arange(0, H).view(1, 1, H, 1).expand(1, 1, H, W)

        grid = torch.cat((xx, yy), 1).float()

        if Input.is_cuda:
            grid = grid.cuda()

        return grid

    def warp(self, x, BM_d):
        vgrid = self.grid + BM_d # [N2HW] # [(2d+1)^2, 2, H, W]
        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(x.size(3) - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(x.size(2) - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = nn.functional.grid_sample(x, vgrid, mode='bilinear', padding_mode = 'border') #800MB memory occupied (d=2,C=64,H=256,W=256)
        mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
        mask = nn.functional.grid_sample(mask, vgrid) #300MB memory occpied (d=2,C=64,H=256,W=256)

        mask = mask.masked_fill_(mask<0.999,0)
        mask = mask.masked_fill_(mask>0,1)

        return output * mask

    def forward(self,sim_feature, f_map, apply_softmax_on_simfeature = True):
        '''
        Return bilateral cost volume(Set of bilateral correlation map)
        :param sim_feature: Correlation feature based on operating frame's HW (N,D2,H,W)
        :param f_map: Previous frame mask (N,1,H,W)
        :return Correlation Cost: (N,(2d+1)^2,H,W)
        '''
        # feature1 = self.L2normalize(feature1)
        # feature2 = self.L2normalize(feature2)

        B_size,C_size,H_size,W_size = f_map.size()

        if self.grid is None:
            # Initialize first uniform grid
            self.grid = self.UniformGrid(f_map)

        if H_size != self.grid.size(2) or W_size != self.grid.size(3):
            # Update uniform grid to fit on input tensor shape
            self.grid = self.UniformGrid(f_map)


        # Displacement volume (N,(2d+1)^2,2,H,W) d = (i,j) , i in [-md,md] & j in [-md,md]
        D_vol = self.d.view(self.range, 2, 1, 1).expand(-1, -1, H_size, W_size)  # [(2d+1)^2, 2, H, W]

        if apply_softmax_on_simfeature:
            sim_feature = F.softmax(sim_feature, dim=1)  # B,D^2,H,W
        f_map = self.warp(f_map.transpose(0, 1).expand(self.range,-1,-1,-1), D_vol).transpose(0, 1) # B,D^2,H,W

        f_map = torch.sum(torch.mul(sim_feature, f_map),dim=1, keepdim=True) # B,1,H,W

        return f_map # B,1,H,W
