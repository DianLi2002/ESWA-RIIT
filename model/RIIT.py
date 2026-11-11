import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange

# ==========================================
# Part 1
# ==========================================

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)
    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv3d(dim, hidden_features * 3, kernel_size=(1, 1, 1), bias=bias)
        self.dwconv1 = nn.Conv3d(hidden_features, hidden_features, kernel_size=(3, 3, 3), stride=1, padding=1, groups=hidden_features, bias=bias)
        self.dwconv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=(3, 3), stride=1, dilation=2, padding=2, groups=hidden_features, bias=bias)
        self.dwconv3 = nn.Conv2d(hidden_features, hidden_features, kernel_size=(3, 3), stride=1, dilation=3, padding=3, groups=hidden_features, bias=bias)
        self.project_out = nn.Conv3d(hidden_features, dim, kernel_size=(1, 1, 1), bias=bias)
    
    def forward(self, x):
        x = x.unsqueeze(2)
        x = self.project_in(x)
        x1, x2, x3 = x.chunk(3, dim=1)
        x1 = self.dwconv1(x1).squeeze(2)
        x2 = self.dwconv2(x2.squeeze(2))
        x3 = self.dwconv3(x3.squeeze(2))
        x = F.gelu(x1) * x2 * x3
        x = x.unsqueeze(2)
        x = self.project_out(x)
        return x.squeeze(2)

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=31, dim=32, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv3d(in_c, dim, kernel_size=(3, 3, 3), stride=1, padding=1, bias=bias)
    def forward(self, x):
        x = x.unsqueeze(2)
        x = self.proj(x)
        return x.squeeze(2)

# ==========================================
# Part 2
# ==========================================
class GaussianBlurLayer(nn.Module):
    def __init__(self, channels, kernel_size=7, sigma=3.0):

        super(GaussianBlurLayer, self).__init__()
        
        # 1. create 1D Gaussian kernel
        radius = kernel_size // 2
        x = torch.arange(-radius, radius + 1, dtype=torch.float32)
        
        # Gaussian
        gauss_1d = torch.exp(-(x**2) / (2 * sigma**2))
        gauss_1d = gauss_1d / gauss_1d.sum() # 归一化
        
        # 2. 2D Gaussian kernel (1D @ 1D.T)
        gauss_2d = gauss_1d.unsqueeze(1) @ gauss_1d.unsqueeze(0)
        
        # 3. weight [channels, 1, k, k]
        # NOTE: expand() returns a view whose storage is shared across channels
        # (stride 0). It must be materialised with contiguous(), otherwise
        # `load_state_dict` fails on PyTorch >= 2.0 ("more than one element of
        # the written-to tensor refers to a single memory location").
        kernel = gauss_2d.expand(channels, 1, kernel_size, kernel_size).contiguous()
        
        # 4. conv layer (Depthwise)
        self.conv = nn.Conv2d(in_channels=channels, out_channels=channels,
                              kernel_size=kernel_size, stride=1, 
                              padding=kernel_size//2,
                              groups=channels,        
                              bias=False)
        
        # 5. Freeze the weights
        self.conv.weight.data = kernel
        self.conv.weight.requires_grad = False 

    def forward(self, x):
        return self.conv(x)

class IlluminationStream(nn.Module):
    def __init__(self, inp_channels, base_dim):
        super(IlluminationStream, self).__init__()
        
        self.enc1 = nn.Sequential(
            nn.Conv2d(inp_channels, base_dim, kernel_size=3, padding=1, stride=1), # Downsample
            nn.InstanceNorm2d(base_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_dim, base_dim * 2, kernel_size=3, padding=1, stride=2), # Downsample
            nn.InstanceNorm2d(base_dim * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_dim * 2, base_dim * 4, kernel_size=3, padding=1, stride=2), # Downsample
            nn.InstanceNorm2d(base_dim * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        # x: [B, C, H, W]
        f1 = self.enc1(x) # -> [B, dim, H, W]
        f2 = self.enc2(f1) # -> [B, 2*dim, H/2, W/2]
        f3 = self.enc3(f2) # -> [B, 4*dim, H/4, W/4]
        return f1, f2, f3


class SpectralCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=4, bias=False):
        super(SpectralCrossAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.to_q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.to_k = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.to_v = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x_r, x_i):
        b, c, h, w = x_r.shape
        
        q = self.to_q(x_r)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        k = self.to_k(x_i)
        v = self.to_v(x_i)
        
        # [B, Heads, C/Heads, N_reduced]
        k = rearrange(k, 'b (head c) h w -> b head (h w) c', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head (h w) c', head=self.num_heads)
        
        q = F.normalize(q, dim=-1) # [B, H, C_dim, N_q]
        k = F.normalize(k, dim=-2) # [B, H, N_k, C_dim] (注意 dim=-2, 因为k还没转置)
            
        # (q @ k) -> [B, H, N_q, N_k] -> OOM Source
        attn = (q @ k) * self.temperature
        attn = attn.softmax(dim=-1)
        # (attn @ v.T) -> [B, H, N_q, C_dim]
        out = (attn @ v.transpose(-2, -1))

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        
        return out

class SpatialCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=4, bias=False):
        super(SpatialCrossAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.to_q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.to_k = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.to_v = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x_r, x_i):
        b, c, h, w = x_r.shape   

        q_s = self.to_q(x_r)
        k_s = self.to_k(x_i)
        v_s = self.to_v(x_i)

        q_s = rearrange(q_s, 'b (head c) h w -> b head (h w) c', head=self.num_heads)
        k_s = rearrange(k_s, 'b (head c) h w -> b head (h w) c', head=self.num_heads)
        v_s = rearrange(v_s, 'b (head c) h w -> b head (h w) c', head=self.num_heads)

        # Linear Attention Kernel: ELU + 1
        q_s = F.elu(q_s) + 1
        k_s = F.elu(k_s) + 1

        # Q @ (K.T @ V)
        # 1. Context Matrix: [B, Head, C/H, C/H]
        context = k_s.transpose(-2, -1) @ v_s 
        
        # 2. Normalization Factor: Q @ sum(K)
        k_sum = k_s.sum(dim=-2, keepdim=True) # [B, Head, 1, C/H]
        denom = q_s @ k_sum.transpose(-2, -1) # [B, Head, N, 1]

        # 3. Output: Q @ Context / Denom
        out_spatial = (q_s @ context) / (denom + 1e-6)
        out_spatial = rearrange(out_spatial, 'b head (h w) c -> b (head c) h w', h=h, w=w)
        out_spatial = self.project_out(out_spatial)

        return out_spatial


class SpectralAttention(nn.Module):
    def __init__(self, dim, num_heads=4, bias=False):
        super(SpectralAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        
        # Q, K, V
        self.to_qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        
        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape to: [B, Heads, HW, C/Heads]
        q = rearrange(q, 'b (head c) h w -> b head (h w) c', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head (h w) c', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head (h w) c', head=self.num_heads)
        
        # Normalize
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        
        # Attention: [B, Heads, C/Heads, C/Heads]
        q = q.transpose(-2, -1) 
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        
        # Attention Map: (C', HW) @ (HW, C') -> (C', C')
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        
        # Output: (C', C') @ (C', HW) -> (C', HW)
        out = attn @ v
        
        # Reshape back
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        
        out = self.project_out(out)
        return out

class DualStreamBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(DualStreamBlock, self).__init__()
        
        # 1. Cross Attention Stream Interaction
        self.norm_cross = LayerNorm(dim, LayerNorm_type)
        self.spectrtral_cross_attn = SpectralCrossAttention(dim, num_heads=num_heads, bias=bias)
        self.spatial_cross_attn = SpatialCrossAttention(dim, num_heads=num_heads, bias=bias)

        self.alpha = nn.Parameter(torch.zeros(1))

        # 2. Spectral Self Attention
        self.norm_self = LayerNorm(dim, LayerNorm_type)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.Conv2d(dim, dim, 5, padding=2, groups=dim), # Depthwise spatial context
            nn.Conv2d(dim, dim, 1)
        ) 
        self.spectral_attn = SpectralAttention(dim, num_heads=num_heads, bias=bias)
        self.beta = nn.Parameter(torch.zeros(1))

        # 3. Feed Forward
        self.norm_ffn = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x_r, x_i): 
        # Step 1: Cross Attention (Interaction)
        if x_i is not None:
            x_spectral_cross = self.spectrtral_cross_attn(self.norm_cross(x_r), x_i)
            x_spatial_cross = self.spatial_cross_attn(self.norm_cross(x_r), x_i)
        
            x_r = x_r + x_spectral_cross + self.alpha * x_spatial_cross
        
        # Step 2: Spectral + Spatial
        residual = x_r
        x_norm = self.norm_self(x_r)
        x_spatial = self.spatial_conv(x_norm)
        x_spectral = self.spectral_attn(x_norm)
        x_r = residual + x_spatial + self.beta * x_spectral
        
        # Step 3: FFN
        x_r = x_r + self.ffn(self.norm_ffn(x_r))
        
        return x_r

class ResizeConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResizeConv, self).__init__()
        self.body = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.body(x)
# ==========================================
# Part 3: RIIT Main Model
# ==========================================

class RIIT(nn.Module):
    def __init__(self, inp_channels=31, out_channels=31, dim=64, 
                 num_blocks=[2, 2, 2], 
                 heads=[2, 4, 8], 
                 ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'):
        super(RIIT, self).__init__()

        self.gaussian_blur = GaussianBlurLayer(channels=inp_channels, kernel_size=9, sigma=3.0)
        # 1. Illumination Stream (Stream B)
        self.illum_stream = IlluminationStream(inp_channels, dim)

        # 2. Reflectance Stream (Stream A) - Main U-Net
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        
        # Encoder Level 1
        self.encoder1 = nn.ModuleList([
            DualStreamBlock(dim, heads[0], ffn_expansion_factor, bias, LayerNorm_type)
            for _ in range(num_blocks[0])
        ])
        
        self.down_conv1 = nn.Conv2d(dim, dim*2, kernel_size=3, stride=2, padding=1)

        # Encoder Level 2
        self.encoder2 = nn.ModuleList([
            DualStreamBlock(dim*2, heads[1], ffn_expansion_factor, bias, LayerNorm_type)
            for _ in range(num_blocks[1])
        ])
        
        self.down_conv2 = nn.Conv2d(dim*2, dim*4, kernel_size=3, stride=2, padding=1)

        # Encoder Level 3 (Bottleneck)
        self.encoder3 = nn.ModuleList([
            DualStreamBlock(dim*4, heads[2], ffn_expansion_factor, bias, LayerNorm_type)
            for _ in range(num_blocks[2])
        ])
        
        # Decoder
        self.up3_2 = ResizeConv(dim*4, dim*2)
        self.reduce_2 = nn.Conv2d(dim*4, dim*2, 1)
        
        self.decoder2 = nn.ModuleList([
            DualStreamBlock(dim*2, heads[1], ffn_expansion_factor, bias, LayerNorm_type)
            for _ in range(num_blocks[1])
        ])
        
        self.up2_1 = ResizeConv(dim*2, dim)
        self.reduce_1 = nn.Conv2d(dim*2, dim, 1)
        
        self.decoder1 = nn.ModuleList([
            DualStreamBlock(dim, heads[0], ffn_expansion_factor, bias, LayerNorm_type)
            for _ in range(num_blocks[0])
        ])
        
        self.refinement = (
            DualStreamBlock(dim=dim, num_heads=1, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        )

        self.output_conv = nn.Conv2d(dim, out_channels, 3, padding=1)

    def forward(self, inp_img):
        low_freq_img = self.gaussian_blur(inp_img)

        illu_f1, illu_f2, illu_f3 = self.illum_stream(low_freq_img)

        x = self.patch_embed(inp_img) # [B, dim, H, W]

        # Encoder Level 1 (with Interaction)
        for blk in self.encoder1:
            x = blk(x, illu_f1) 
        enc1_out = x
        
        x = self.down_conv1(x) # [B, 2*dim, H/2, W/2]
        
        # Encoder Level 2 (with Interaction)
        for blk in self.encoder2:
            x = blk(x, illu_f2)
        enc2_out = x
        
        x = self.down_conv2(x) # [B, 4*dim, H/4, W/4]
        
        # Encoder Level 3 (Bottleneck) (with Interaction)
        for blk in self.encoder3:
            x = blk(x, illu_f3)
            
        # --- Decoder ---
        
        # Up 3->2
        x = self.up3_2(x) # [B, 2*dim, H/2, W/2]
        x = torch.cat([x, enc2_out], dim=1)
        x = self.reduce_2(x)
        
        for blk in self.decoder2:
            x = blk(x, None)
            
        # Up 2->1
        x = self.up2_1(x) # [B, dim, H, W]
        x = torch.cat([x, enc1_out], dim=1)
        x = self.reduce_1(x)
        
        for blk in self.decoder1:
            x = blk(x, None)

        x = self.refinement(x, None)

        # Output
        out_res = self.output_conv(x)
        return out_res + inp_img

# test code
if __name__ == "__main__":
    x = torch.randn(1, 31, 256, 256)
    model = RIIT(inp_channels=31, out_channels=31, dim=64)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    # calculate parameters and FLOPs
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params / 1e6:.2f} M")

    from thop import profile, clever_format
    MACs, params = profile(model, inputs=(x,), verbose=True)

    MACs, params = clever_format([MACs, params], '%.3f')

    print(f"FLOPs:{MACs}, Params:{params}")
    print("-------------------------------------------")