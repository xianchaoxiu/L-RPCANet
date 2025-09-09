import torch
import torch.nn as nn
from einops import rearrange, repeat
import math
import torch.nn.functional as F

import numpy as np

__all__ = ['LRPCANet']

class LRPCANet(nn.Module):
    def __init__(self, stage_num=6, rlayers=6, slayers=6, llayers=3, mlayers=3, b_channel=16, channel=32, mode='train'):
        super(LRPCANet, self).__init__()
        self.stage_num = stage_num
        self.decos = nn.ModuleList()
        self.mode = mode
        for _ in range(stage_num):
            self.decos.append(DecompositionModule(slayers=slayers, llayers=llayers, rlayers=rlayers, mlayers=mlayers, b_channel=b_channel, channel=channel))

        # 迭代循环初始化参数
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, D):
        T = torch.zeros(D.shape).to(D.device)
        N = torch.zeros(D.shape).to(D.device)
        for i in range(self.stage_num):
            D, T = self.decos[i](D, T, N)
        if self.mode == 'train':
            return D, T
        else:
            return T

class DecompositionModule(nn.Module):
    def __init__(self, rlayers=6, slayers=6, llayers=3, mlayers=3, b_channel=16, channel=32):
        super(DecompositionModule, self).__init__()
        self.lowrank = LowrankModule(channel=channel, b_channel=b_channel, layers=llayers)
        self.sparse = SparseModule(channel=channel, b_channel=b_channel, layers=slayers)
        self.residual = ResidualBlock(channel=channel, b_channel=b_channel, layers=rlayers)
        self.merge = MergeModule(channel=channel, layers=mlayers)

    def forward(self, D, T, N):
        B = self.lowrank(D, T, N)
        T = self.sparse(D, B, T, N)
        N = self.residual(D, B, T, N)
        D = self.merge(B, T, N)
        return D, T

class SEModule(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(SEModule, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(channels, channels // reduction_ratio, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction_ratio, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.squeeze(x)
        s = self.excitation(z)
        return x * s


class LowrankModule(nn.Module):
    def __init__(self, in_channel=1, b_channel=16, channel=32, layers=3):
        super(LowrankModule, self).__init__()

        convs = [nn.Conv2d(in_channel, b_channel, kernel_size=1, padding=0, stride=1), 
                 nn.BatchNorm2d(b_channel),
                 nn.ReLU(True)]
                 
        for i in range(layers):
            convs.append(nn.Conv2d(b_channel, b_channel, kernel_size=3, padding=1, stride=1))
            convs.append(nn.BatchNorm2d(b_channel))
            convs.append(nn.ReLU(True))

        convs.append(nn.Conv2d(b_channel, channel, kernel_size=3, padding=1, stride=1))
        self.convs = nn.Sequential(*convs)
        self.channel_attention = SEModule(channel)             
        self.final_conv = nn.Conv2d(channel, in_channel, kernel_size=3, padding=1, stride=1)
       
    def forward(self, D, T, N):
        x = D - T - N
        F = self.convs(x)
        F = self.channel_attention(F)
        B = x + self.final_conv(F)
        return B
        
class SparseModule(nn.Module):
    def __init__(self, in_channel=1, b_channel=16, channel=32, layers=6):
        super(SparseModule, self).__init__()
        convs = [nn.Conv2d(1, b_channel, kernel_size=1, padding=0, stride=1), 
                 nn.ReLU(True)]
                 
        for i in range(layers):
            convs.append(nn.Conv2d(b_channel, b_channel, kernel_size=3, padding=1, stride=1))
            convs.append(nn.ReLU(True))
            
        convs.append(nn.Conv2d(b_channel, channel, kernel_size=3, padding=1, stride=1))           
        self.convs = nn.Sequential(*convs)
        self.channel_attention = SEModule(channel)      
        self.final_conv = nn.Conv2d(channel, in_channel, kernel_size=3, padding=1, stride=1)
        self.epsilon = nn.Parameter(torch.Tensor([0.01]), requires_grad=True)

        
    def forward(self, D, B, T, N):
        x = T + D - B - N
        F = self.convs(x)
        F = self.channel_attention(F)   
        T = x - self.epsilon * self.final_conv(F)
        return T

class ResidualBlock(nn.Module):
    def __init__(self, in_channel=1, b_channel=16, channel=32, layers=6):
        super(ResidualBlock, self).__init__()
        convs = [nn.Conv2d(in_channel, b_channel, kernel_size=3, padding=1, stride=1),
                 nn.ReLU(True)]
                 
        for i in range(layers):
            convs.append(nn.Conv2d(b_channel, b_channel, kernel_size=3, padding=1, stride=1))
            convs.append(nn.ReLU(True))
            
        convs.append(nn.Conv2d(b_channel, channel, kernel_size=3, padding=1, stride=1))  
        self.convs = nn.Sequential(*convs)
        self.channel_attention = SEModule(channel)       
        self.final_conv = nn.Conv2d(channel, in_channel, kernel_size=3, padding=1, stride=1)
        self.sigma = nn.Parameter(torch.Tensor([0.01]), requires_grad=True)

    def forward(self, D, B, T, N):
        x = N + D - B - T
        F = self.convs(x)
        F = self.channel_attention(F)  
        N = x - self.sigma * self.final_conv(F)
        return N

class MergeModule(nn.Module):
    def __init__(self, in_channel=1, channel=32, layers=3):
        super(MergeModule, self).__init__()
        convs = [nn.Conv2d(in_channel, channel, kernel_size=3, padding=1, stride=1),
                 nn.BatchNorm2d(channel),
                 nn.ReLU(True)]
                 
        for i in range(layers):
            convs.append(nn.Conv2d(channel, channel, kernel_size=3, padding=1, stride=1))
            convs.append(nn.BatchNorm2d(channel))
            convs.append(nn.ReLU(True))
        
        self.mapping = nn.Sequential(*convs)
        self.channel_attention = SEModule(channel)  
        self.final_conv = nn.Conv2d(channel, in_channel, kernel_size=3, padding=1, stride=1)    


    def forward(self, B, T, N):
        x = B + T + N
        F = self.mapping(x)
        F = self.channel_attention(F)
        D = self.final_conv(F)
        return D

