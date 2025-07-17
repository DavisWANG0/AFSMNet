import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import ops
from basicsr.utils.registry import ARCH_REGISTRY


# Layer Norm
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

# CCM
class CCM(nn.Module):
    def __init__(self, dim, growth_rate=2.):
        super().__init__()
        hidden_dim = int(dim * growth_rate)

        self.project_in = nn.Conv2d(dim,hidden_dim*2,1,1,0)
        self.dwconv = nn.Conv2d(hidden_dim*2,hidden_dim*2,3,1,1,groups=hidden_dim*2)
        self.project_out = nn.Conv2d(hidden_dim,dim,1,1,0)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class SimpleChannelAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim,dim,1,1,0,bias=True),
            nn.Sigmoid())

    def forward(self, x):
        return self.sca(x)

# SAFM
class SAFM(nn.Module):
    def __init__(self, dim, branch_ratio=0.25):
        super().__init__()

        gc = int(dim * branch_ratio)
        self.conv_real_1 = nn.Conv2d(gc, gc, kernel_size=1, stride=1, padding=0, bias=True)
        self.scale = 0.02
        self.conv_real_1.weight = nn.Parameter(self.scale*torch.randn_like(self.conv_real_1.weight))
        self.conv_real_1.bias = nn.Parameter(self.scale*torch.randn_like(self.conv_real_1.bias))
        self.act = nn.GELU()
        self.conv_real_2 = nn.Conv2d(gc, gc, kernel_size=1)

        self.conv = nn.Sequential(nn.Conv2d(gc,gc,1,1,0),
                                  nn.GELU(),
                                  nn.Conv2d(gc,gc,1,1,0))
        
        self.split_indexes = (gc, gc, dim - 2 * gc)
        self.sca = SimpleChannelAttention(dim)

        self.refine = nn.Conv2d(dim, dim, 1, 1, 0)


    def forward(self, x): 
        B, C, H, W = x.shape

        score = self.sca(x)

        _, sorted_indices = torch.sort(score.squeeze(-1).squeeze(-1), dim=1, descending=True)
        x_reordered = torch.gather(x, dim=1, index=sorted_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W))

        x_1, x_2, x_3 = torch.split(x_reordered, self.split_indexes, dim=1)

        x_1 = torch.fft.fft2(x_1, dim=(1), norm="ortho")
        x_1 = self.act(self.conv_real_1(x_1.real))
        x_1 = self.conv_real_2(x_1)
        x_1 = torch.fft.ifft2(x_1, dim=(1), norm="ortho").to(x.dtype)

        x_2 = self.conv(x_2)

        x = torch.cat([x_1, x_2, x_3], dim=1)

        original_order = torch.argsort(sorted_indices, dim=1)
        x = torch.gather(x_reordered, dim=1, index=original_order.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W))

        return self.refine(x)


class AttBlock(nn.Module):
    def __init__(self, dim, branch_ratio, growth_rate):
        super().__init__()

        self.norm1 = LayerNorm(dim) 
        self.norm2 = LayerNorm(dim) 

        # Multiscale Block
        self.safm = SAFM(dim, branch_ratio) 
        # Feedforward layer
        self.ccm = CCM(dim, growth_rate) 

    def forward(self, x):
        x = self.safm(self.norm1(x)) + x
        x = self.ccm(self.norm2(x)) + x
        return x
        
        
@ARCH_REGISTRY.register()
class AFSMNet(nn.Module):
    def __init__(self, dim, n_blocks=16, branch_ratio=0.4, growth_rate=2.66, upscaling_factor=4):
        super().__init__()
        self.to_feat = nn.Conv2d(3, dim, 3, 1, 1)

        self.feats = nn.Sequential(*[AttBlock(dim, branch_ratio, growth_rate) for _ in range(n_blocks)])

        self.to_img = nn.Sequential(
            nn.Conv2d(dim, 3 * upscaling_factor**2, 3, 1, 1),
            nn.PixelShuffle(upscaling_factor)
        )

    def forward(self, x):
        x = self.to_feat(x)
        x = self.feats(x) + x
        x = self.to_img(x)
        return x



if __name__== '__main__':
    #############Test Model Complexity #############
    from fvcore.nn import flop_count_table, FlopCountAnalysis, ActivationCountAnalysis  

    # for x2 SR
    x = torch.randn(1, 3, 640, 360)
    model = AFSMNet(dim=36, n_blocks=16, branch_ratio=0.4, growth_rate=2.66, upscaling_factor=2)

    # for x3 SR
    # x = torch.randn(1, 3, 427, 240)
    # model = AFSMNet(dim=36, n_blocks=16, branch_ratio=0.4, growth_rate=2.66, upscaling_factor=3)

    # for x4 SR
    # x = torch.randn(1, 3, 320, 180)
    # model = AFSMNet(dim=36, n_blocks=16, branch_ratio=0.4, growth_rate=2.66, upscaling_factor=4)

    print(model)
    print(f'params: {sum(map(lambda x: x.numel(), model.parameters()))}')
    print(flop_count_table(FlopCountAnalysis(model, x), activations=ActivationCountAnalysis(model, x)))
    output = model(x)
    print(output.shape)
