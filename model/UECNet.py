# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 16:14:37 2021

@author: Administrator
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch, math
import numbers
import clip
import torchvision.transforms as T
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import init
from torch.nn.modules.utils import _pair
from torchvision.ops.deform_conv import deform_conv2d
from einops import rearrange
from model.Prompt_learning import Prompts, TextEncoder
from timm.layers import DropPath

class UECNet(nn.Module):
    def __init__(self, channel=32, prompt_path=None, device='cuda'):
        super(UECNet, self).__init__()
        self.norm = lambda x: (x - 0.5) / 0.5
        self.denorm = lambda x: (x + 1) / 2

        self.class_text = [ "Natural lighting, balanced contrast region.",
                            "White glare and intense highlight region.",
                            "Slightly overexposed region.",
                            "Underexposed and dim region.",
                            "Completely dark and black region."]
        self.clip_model = CLIPTextOnly("ViT-B/32", device=device)
        self.exposure_map = Exposure_Map(self.clip_model, prompt_path, self.class_text)
        # self.guide_textf = Guidetext_Generator(self.clip_model, self.guide_text)

        self.prompt_gen1 = Prompt_Generator(in_channels=3)
        self.in_conv = nn.Conv2d(3, channel, kernel_size=1, stride=1, padding=0, bias=False)

        self.down = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.enc1 = TransformerBlock(channel, guide=True)
        self.econv1_2 = nn.Conv2d(channel, 2*channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.enc2 = TransformerBlock(channel*2)
        self.econv2_3 = nn.Conv2d(2*channel, 4*channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.enc3 = TransformerBlock(channel*4)
        self.econv3_4 = nn.Conv2d(4*channel, 8*channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.enc4 = TransformerBlock(channel*8)

        self.prompt_gen2 = Prompt_Generator(8 * channel)
        self.middle_1 = TransformerBlock(channel*8, guide=True)
        self.middle_2 = TransformerBlock(channel*8)

        self.prompt_gen3 = Prompt_Generator(8 * channel)
        self.dec4 = TransformerBlock(channel*8, guide=True)
        self.dconv4_3 = nn.Conv2d(8*channel, 4*channel, kernel_size=1, stride=1, padding=0, bias=False)  # 128 64
        self.dec3 = TransformerBlock(channel*4)
        self.dconv3_2 = nn.Conv2d(4*channel, 2*channel, kernel_size=1, stride=1, padding=0, bias=False)  # 64 32
        self.dec2 = TransformerBlock(channel*2)
        self.dconv2_1 = nn.Conv2d(2*channel, channel, kernel_size=1, stride=1, padding=0, bias=False)  # 32 16
        self.dec1 = TransformerBlock(channel)

        self.out_conv = nn.Conv2d(channel, 3, kernel_size=1, stride=1, padding=0, bias=False)

    def _up(self, x, y):
        _, _, H0, W0 = y.size()
        return F.interpolate(x, size=(H0, W0), mode='bilinear')

    def forward(self, x):
        exposure_txtf, exposure_map = self.exposure_map(x)
        # guide_textf = self.guide_textf()
        prompt1 = self.prompt_gen1(x, exposure_map, exposure_txtf)
        x_in = self.in_conv(self.norm(x)) # channel:3-->32, H-->H, W-->W
        # Encoder
        out_enc_1 = self.enc1(x_in, prompt1)
        out_enc_2 = self.enc2(self.econv1_2(self.down(out_enc_1)), None) # channel:32-->64, H-->H/2, W-->W/2
        out_enc_3 = self.enc3(self.econv2_3(self.down(out_enc_2)), None) # channel:64-->128, H/2-->H/4, W/2-->W/4
        out_enc_4 = self.enc4(self.econv3_4(self.down(out_enc_3)), None) # channel:128-->256, H/4-->H/8, W/4-->W/8
        # Latent
        prompt2 = self.prompt_gen2(out_enc_4, exposure_map, exposure_txtf)
        middle_1 = self.middle_1(out_enc_4, prompt2)
        middle_2 = self.middle_2(middle_1, None)
        # Decoder
        prompt3= self.prompt_gen3(middle_2, exposure_map, exposure_txtf)
        out_dec_4 = self.dec4(middle_2+out_enc_4, prompt3)
        out_dec_3 = self.dconv4_3(self._up(out_dec_4, out_enc_3)) # channel:256-->128, H/8-->H/4, W/8-->W/4
        out_dec_3 = self.dec3(out_dec_3+out_enc_3, None) # Add, not concat
        out_dec_2 = self.dconv3_2(self._up(out_dec_3, out_enc_2)) # channel:128-->64, H/4-->H/2, W/4-->W/2
        out_dec_2 = self.dec2(out_dec_2+out_enc_2, None)
        out_dec_1 = self.dconv2_1(self._up(out_dec_2, out_enc_1)) # channel:64-->32, H/2-->H, W/2-->W
        out_dec_1 = self.dec1(out_dec_1+out_enc_1, None)

        out = self.out_conv(out_dec_1) + x

        return out

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

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


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CLIPTextOnly(nn.Module):
    def __init__(self, model_name="ViT-B/32", device="cuda"):
        super().__init__()
        full_model, _ = clip.load(model_name, device=device)

        # 保留文本部分
        self.token_embedding = full_model.token_embedding
        self.positional_embedding = full_model.positional_embedding
        self.transformer = full_model.transformer
        self.ln_final = full_model.ln_final
        self.text_projection = full_model.text_projection
        self.encode_text = full_model.encode_text
        # 删除图像部分
        del full_model.visual

        for p in self.parameters():
            p.requires_grad = False

        self.float()


##########################################################################
##
class Exposure_Map(nn.Module):
    def __init__(self, clip_model, model_path, class_text, device='cuda'):
        super().__init__()
        self.model = clip_model.float()
        self.image_transform = T.Compose([
            T.Resize((224, 224)),
            T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                        std=(0.26862954, 0.26130258, 0.27577711))
        ])
        # 创建Prompts实例
        self.prompt_model = Prompts(clip_model=self.model, initials=class_text, train_backbone=False).to(device)
        state_dict = torch.load(model_path, map_location=device)
        self.prompt_model.load_state_dict(state_dict)
        self.prompt_model.eval()
        for name, param in self.prompt_model.named_parameters():
            param.requires_grad = False

        # Exposure text features
        self.embedding_prompt = self.prompt_model.embedding_prompt
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in class_text]).to(device) #使用和训练相同的text得到token_id
        self.text_encoder = TextEncoder(self.model)
        with torch.no_grad():
            self.text_features = self.text_encoder(self.embedding_prompt, tokenized_prompts)
            self.text_features = F.normalize(self.text_features, dim=-1)
            self.text_features = self.prompt_model.text_feat_compressor(self.text_features)

    def forward(self, image):
        B, C, H, W = image.shape
        image = self.image_transform(image)
        patch_size = 32
        stride = 32
        # Crop to patch：[B, C, H, W] -> [B*num_patches, C, patch_size, patch_size]
        patches = image.unfold(2, patch_size, stride).unfold(3, patch_size, stride)  # [B, C, nH, nW, ps, ps]
        B_, C_, nH, nW, ps, _ = patches.shape
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()  # [B, nH, nW, C, ps, ps]
        patches = patches.view(-1, C_, ps, ps)  # [B*nH*nW, C, ps, ps]
        # Batch encode using CLIP
        with torch.no_grad():
            image_feat = self.prompt_model.image_backbone(patches)[0]
            image_feat = self.prompt_model.img_embed(image_feat)
            image_feat = image_feat.view(image_feat.size(0), -1)
            image_feat = self.prompt_model.img_fc(image_feat)
            image_feat = F.normalize(image_feat, dim=-1)

        sim = image_feat @ self.text_features.T  # → [B*nH*nW, 512]@[512,5]-->[B*nH*nW, 5]
        sim = sim.view(B, nH, nW, 5).permute(0, 3, 1, 2)  # → [B, 5, nH, nW]
        exposure_rito_map = F.softmax(sim/0.07, dim=1)  # soft label，每个 patch 属于各类别的概率-->[B, 5, nH, nW]

        return self.text_features, exposure_rito_map   # [N, 128, H, W], [N, 5, H, W]

############可学习的指导语义#############
# class Guidetext_Generator(nn.Module):
#     def __init__(self, clip_model, guide_text, device='cuda'):
#         super().__init__()
#         self.model = clip_model.float()
#         self.text_encoder = TextEncoder(self.model)
#         self.tokenized_prompts_g = torch.cat([clip.tokenize(p) for p in guide_text]).to(device)
#         self.prompts_guide = nn.Parameter(
#             self.model.token_embedding(self.tokenized_prompts_g).to(torch.float32).requires_grad_()).cuda()
#         self.text_feat_compressor = nn.Sequential(
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Linear(256, 128),
#             nn.ReLU()
#         )
#
#     def forward(self):
#         guide_text_f = self.text_encoder(self.prompts_guide, self.tokenized_prompts_g)
#         guide_text_f = F.normalize(guide_text_f, dim=-1)
#         guide_text_f = self.text_feat_compressor(guide_text_f)
#         return guide_text_f
############固定的的指导语义#############
class Guidetext_Generator(nn.Module):
    def __init__(self, clip_model, guide_text, device='cuda'):
        super().__init__()
        self.model = clip_model.float()
        # self.text_encoder = TextEncoder(self.model)
        self.tokenized_prompts_g = torch.cat([clip.tokenize(p) for p in guide_text]).to(device)
        with torch.no_grad():
            self.guide_text_f = self.model.encode_text(self.tokenized_prompts_g)
        # self.prompts_guide = nn.Parameter(
        #     self.model.token_embedding(self.tokenized_prompts_g).to(torch.float32).requires_grad_()).cuda()
        self.text_feat_compressor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

    def forward(self):
        guide_text_f = F.normalize(self.guide_text_f, dim=-1)
        guide_text_f = self.text_feat_compressor(guide_text_f)
        return guide_text_f

class Prompt_Generator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//2, 5, kernel_size=1)
        )
    def forward(self, img_feat, exposure_rito_map, guide_text_f):
        B, C, H, W = img_feat.shape
        if C > 3:
            modulator = self.mlp(img_feat)  # [B, 5, H, W]
            modulator = torch.sigmoid(modulator)
            exposure_rito_map = F.interpolate(exposure_rito_map, size=(H, W), mode='bicubic', align_corners=False)
            exposure_rito_map = exposure_rito_map * modulator
        else:
            exposure_rito_map = F.interpolate(exposure_rito_map, size=(H, W), mode='bicubic', align_corners=False)
        guide_text_f = guide_text_f.view(1, 5, 128, 1, 1)
        exposure_rito_map = exposure_rito_map.unsqueeze(2)  # [B, 5, 1, nH, nW]
        weighted_features = exposure_rito_map * guide_text_f
        exposure_semantic_map = weighted_features.sum(dim=1)
        return exposure_semantic_map


class ATMOp(nn.Module):
    def __init__(
            self, in_chans, out_chans, stride: int = 1, padding: int = 0, dilation: int = 1,
            bias: bool = True, dimension: str = ''
    ):
        super(ATMOp, self).__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.dimension = dimension

        self.weight = nn.Parameter(torch.empty(out_chans, in_chans, 1, 1))  # kernel_size = (1, 1)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_chans))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input, offset):
        """
        ATM along one dimension, the shape will not be changed
        input: [B, C, H, W]
        offset: [B, C, H, W]
        """
        B, C, H, W = input.size()
        offset_t = torch.zeros(B, 2 * C, H, W, dtype=input.dtype, layout=input.layout, device=input.device)
        if self.dimension == 'w':
            offset_t[:, 1::2, :, :] += offset
        elif self.dimension == 'h':
            offset_t[:, 0::2, :, :] += offset
        else:
            raise NotImplementedError(f"{self.dimension} dimension not implemented")
        return deform_conv2d(
            input, offset_t, self.weight, self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation
        )


class ATMLayer(nn.Module):
    def __init__(self, dim, proj_drop=0.):
        super().__init__()
        self.dim = dim

        # self.atm_c = nn.Linear(dim, dim, bias=False)
        self.atm_h = ATMOp(dim, dim, dimension='h')
        self.atm_w = ATMOp(dim, dim, dimension='w')
        self.atm_local = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim)

        self.fusion = Mlp(dim, dim // 4, dim * 3)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, offset):
        """
        x: [B, H, W, C]
        offsets: [B, 2C, H, W]
        """
        B, H, W, C = x.shape
        assert offset.shape == (B, 2 * C, H, W), f"offset shape not match, got {offset.shape}"
        w = self.atm_w(x.permute(0, 3, 1, 2), offset[:, :C, :, :].contiguous()).permute(0, 2, 3, 1).contiguous()
        h = self.atm_h(x.permute(0, 3, 1, 2), offset[:, C:, :, :].contiguous()).permute(0, 2, 3, 1).contiguous()
        # c = self.atm_c(x)
        # print("======x_local", x.size())
        c = self.atm_local(x.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()

        a = (w + h + c).permute(0, 3, 1, 2).contiguous().flatten(2).mean(2)

        a = self.fusion(a).reshape(B, C, 3).permute(2, 0, 1).contiguous().softmax(dim=0).unsqueeze(2).unsqueeze(2)

        x = w * a[0] + h * a[1] + c * a[2]

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class DynamicActiveBlock(nn.Module):
    def __init__(self, dim, p_dim, mlp_ratio=4., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, share_dim=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.atm = ATMLayer(dim)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.share_dim = share_dim
        self.conv1 = nn.Linear(dim + p_dim, dim)

        self.offset_layer = nn.Sequential(
            norm_layer(dim + p_dim),
            nn.Linear(dim + p_dim, dim * 2 // self.share_dim)
        )

    def forward(self, x, prompt_x):
        """
        :param x: [B, C, H, W]
        :param offset: [B, 2C, H, W]
        """
        x = x.permute(0, 2, 3, 1).contiguous()# -> [B, H, W, C]
        prompt_x = prompt_x.permute(0, 2, 3, 1).contiguous()
        x_p = torch.cat([x, prompt_x], dim=-1)
        offset = self.offset_layer(x_p).repeat_interleave(self.share_dim, dim=-1).permute(0, 3, 1, 2).contiguous()  # [B, H, W, 2C/S] -> [B, 2C, H, W]
        x_p = self.conv1(x_p)
        x = x + self.drop_path(self.atm(self.norm1(x_p), offset))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x.permute(0, 3, 1, 2).contiguous()


class Self_Attention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 bias):
        super(Self_Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class FeedForward(nn.Module):
    def __init__(self,
                 dim,
                 ffn_expansion_factor,
                 bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',
                 guide=False):
        super(TransformerBlock, self).__init__()

        # self.norm1 = LayerNorm(dim, LayerNorm_type)
        # self.cross_attn = Cross_Attention(dim, num_heads, bias)
        self.guide = guide
        if self.guide:
            self.offset_block = DynamicActiveBlock(dim, 128)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.attn = Self_Attention(dim, num_heads, bias)
        self.norm3 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x, emb=None):
        if emb is not None:
            x = self.offset_block(x, emb)
        x = x + self.attn(self.norm2(x))
        x = x + self.ffn(self.norm3(x))
        return x


if __name__ == '__main__':
    device = 'cuda'
    # clip_model, _ = clip.load("ViT-B/32", device=device)
    net = LightRestore(prompt_path='../ckpts/best_valacc_class5_06271723.pth').to("cuda" if torch.cuda.is_available() else "cpu")
    input = torch.randn(1, 3, 256, 256).to("cuda" if torch.cuda.is_available() else "cpu")
    #----------------------------------------------------------------------------------#
    # from thop import profile, clever_format
    # flops, params = profile(net, inputs=(input,))
    # flops, params = clever_format([flops, params], '%.3f')
    # print(f"运算量：{flops}, 参数量：{params}")
    # ----------------------------------------------------------------------------------#
    from ptflops import get_model_complexity_info
    input_res = (3, 256, 256)
    macs, params = get_model_complexity_info(net, input_res, as_strings=True, print_per_layer_stat=True)
    print(f"模型FLOPs: {macs}")
    print(f"模型参数量: {params}")

    from fvcore.nn import FlopCountAnalysis, parameter_count_table

    # flops = FlopCountAnalysis(net, (input, embedding))
    # print("FLOPs", flops.total() / 1000 ** 3)


