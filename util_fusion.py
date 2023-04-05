#assume we have a FasterNet-style convolution block with partial spatial convolutions and inverted bottleneck using pointwise convolutions
#the term "groups" is here misused as the ratio to determine how much of the T-shaped convolution is 3x3x3 and how much 1x1x1
#note that we use BatchNorm to be able to fuse at inference

import torch
from torch import nn, Tensor
import copy
import torch.nn.functional as F

class Faster(torch.nn.Module):
    """
    adapted from FasterNets but extended with ideas akin to RepVGG 
    """

    def __init__(self,dim_in,dim_out,stride,groups=4):
        super().__init__()
        self.dim_out = dim_out
        self.dim_in = dim_in
        self.groups = groups
        self.stride = stride
        self.conv = nn.Conv3d(dim_in//groups,(dim_out//groups)*2,3,padding=1)
        self.conv1 = nn.Conv3d(dim_in-dim_in//groups,(dim_out-dim_out//groups)*2,1)
        self.norm = nn.BatchNorm3d(dim_out*2)
        self.cout = nn.Conv3d(dim_out*2,dim_out,1)#,bias=False)
        self.relu = nn.LeakyReLU(negative_slope=1e-2,inplace=True)
    def forward(self, x: Tensor) -> Tensor:
        y = self.conv(x[:,:self.dim_in//self.groups])
        z = self.conv1(x[:,self.dim_in//self.groups:])
        yz = self.norm(torch.cat((y,z),1))
        x = self.relu(self.cout(yz))
        if(torch.tensor(self.stride).max()!=1):
            x = nn.functional.max_pool3d(x,self.stride)
        return x
      
#here all the magic happens all the magic:  

class FasterFused(torch.nn.Module):
    """
    fusion or folding of batchnorm *and* subsequent convolution into single operator each.
    """

    def __init__(self,module):
        super().__init__()
        self.dim_out = module.dim_out
        self.dim_in = module.dim_in
        self.groups = module.groups
        self.stride = module.stride
        dim_out = self.dim_out
        dim_in = self.dim_in
        groups = self.groups
        
        #important
        module.eval()
        
        conv = copy.deepcopy(module.conv)
        norm = copy.deepcopy(module.norm)
        cout = copy.deepcopy(module.cout)

        bn_var_rsqrt = torch.rsqrt(norm.running_var.data[:dim_out*2//groups] + 1e-5)
        fused_conv_w = conv.weight.data * (norm.weight.data[:dim_out*2//groups] * bn_var_rsqrt).view(-1,1,1,1,1)
        fused_conv_b = (conv.bias.data - norm.running_mean.data[:dim_out*2//groups]) * bn_var_rsqrt * norm.weight.data[:dim_out*2//4] + norm.bias.data[:dim_out*2//groups]

        fused_all_w = torch.mm(cout.weight.data[:,:dim_out*2//groups].squeeze(),fused_conv_w.reshape(dim_out*2//4,-1)).reshape(dim_out,-1,3,3,3)
        fused_all_b = torch.mm(fused_conv_b.unsqueeze(0),cout.weight.data[:,:dim_out*2//groups].squeeze().t()).squeeze()+.5*cout.bias.data#


        conv1 = copy.deepcopy(module.conv1)

        bn_var_rsqrt = torch.rsqrt(norm.running_var.data[dim_out*2//groups:] + 1e-5)
        fused_conv_w1 = conv1.weight.data * (norm.weight.data[dim_out*2//groups:] * bn_var_rsqrt).reshape(-1,1,1,1,1)
        fused_conv_b1 = (conv1.bias.data - norm.running_mean.data[dim_out*2//groups:]) * bn_var_rsqrt * norm.weight.data[dim_out*2//groups:] + norm.bias.data[dim_out*2//groups:]

        fused_all_w1 = torch.mm(cout.weight.data[:,dim_out*2//groups:].squeeze(),fused_conv_w1.reshape(3*dim_out*2//groups,-1)).reshape(dim_out,-1,1,1,1)
        fused_all_b1 = torch.mm(fused_conv_b1.unsqueeze(0),cout.weight.data[:,dim_out*2//groups:].squeeze().t()).squeeze()+.5*cout.bias.data

        self.conv = nn.Conv3d(dim_in//4,dim_out//4,3,padding=1)
        self.conv.weight.data = fused_all_w
        self.conv.bias.data = fused_all_b
        
        self.conv1 = nn.Conv3d(3*dim_in//4,3*dim_out//4,1)
        self.conv1.weight.data = fused_all_w1
        self.conv1.bias.data = fused_all_b1
        

    def forward(self, x: Tensor) -> Tensor:
        
        dim = self.dim_in
        
        y = self.conv(x[:,:dim//4])
        z = self.conv1(x[:,dim//4:])
        x = F.leaky_relu(y+z,1e-2)
        
        if(torch.tensor(self.stride).max()!=1):
            x = nn.functional.max_pool3d(x,self.stride)
        return x
    
