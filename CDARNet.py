import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from math import sqrt


def conv_2d(inp, oup, kernel_size=3, stride=1, padding=0, groups=1, bias=False, norm=True, act=True):
    conv = nn.Sequential()
    conv.add_module('conv', nn.Conv2d(inp, oup, kernel_size, stride, padding, bias=bias, groups=groups))
    if norm:
        conv.add_module('BatchNorm2d', nn.BatchNorm2d(oup))
    if act:
        conv.add_module('Activation', nn.SiLU())
    return conv


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        # hidden_dim = int(round(inp * expand_ratio))
        hidden_dim = int(round(inp * expand_ratio))
        self.block = nn.Sequential()
        if expand_ratio != 1:
            self.block.add_module('exp_1x1', conv_2d(inp, hidden_dim, kernel_size=1, stride=1, padding=0))
        self.block.add_module('conv_3x3', conv_2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1,
                                                  groups=hidden_dim))
        self.block.add_module('red_1x1', conv_2d(hidden_dim, oup, kernel_size=1, stride=1, padding=0, act=False))
        self.use_res_connect = self.stride == 1 and inp == oup

    def forward(self, x):
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)


class Attention(nn.Module):
    def __init__(self, embed_dim, heads=4, dim_head=8, attn_dropout=0):
        super().__init__()
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.embed_dim = embed_dim
        self.num_heads = heads
        self.scale = dim_head ** -0.5

    def forward(self, x):
        b_sz, S_len, in_channels = x.shape
        # self-attention
        # [N, S, C] --> [N, S, 3C] --> [N, S, 3, h, c] where C = hc
        qkv = self.qkv_proj(x).reshape(b_sz, S_len, 3, self.num_heads, -1)
        # [N, S, 3, h, c] --> [N, h, 3, S, C]
        qkv = qkv.transpose(1, 3).contiguous()
        # [N, h, 3, S, C] --> [N, h, S, C] x 3
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        q = q * self.scale
        # [N h, T, c] --> [N, h, c, T]
        k = k.transpose(-1, -2)
        # QK^T
        # [N, h, S, c] x [N, h, c, T] --> [N, h, S, T]
        attn = torch.matmul(q, k)
        batch_size, num_heads, num_src_tokens, num_tgt_tokens = attn.shape
        attn_dtype = attn.dtype
        attn_as_float = self.softmax(attn.float())
        attn = attn_as_float.to(attn_dtype)
        attn = self.attn_dropout(attn)

        # weighted sum
        # [N, h, S, T] x [N, h, T, c] --> [N, h, S, c]
        out = torch.matmul(attn, v)
        # [N, h, S, c] --> [N, S, h, c] --> [N, S, C]
        out = out.transpose(1, 2).reshape(b_sz, S_len, -1)
        out = self.out_proj(out)

        return out


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, ffn_latent_dim, heads=8, dim_head=8, dropout=0, attn_dropout=0):
        super().__init__()
        self.pre_norm_mha = nn.Sequential(
            nn.LayerNorm(embed_dim, eps=1e-5, elementwise_affine=True),
            Attention(embed_dim, heads, dim_head, attn_dropout),
            nn.Dropout(dropout)
        )
        self.pre_norm_ffn = nn.Sequential(
            nn.LayerNorm(embed_dim, eps=1e-5, elementwise_affine=True),
            nn.Linear(embed_dim, ffn_latent_dim, bias=True),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_latent_dim, embed_dim, bias=True),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Multi-head attention
        x = x + self.pre_norm_mha(x)
        # Feed Forward network
        x = x + self.pre_norm_ffn(x)
        return x


class MobileViTBlockV3(nn.Module):
    def __init__(self, inp, attn_dim, ffn_multiplier, heads, dim_head, attn_blocks, patch_size):
        super(MobileViTBlockV3, self).__init__()
        self.patch_h, self.patch_w = patch_size
        self.patch_area = int(self.patch_h * self.patch_w)

        # local representation
        self.local_rep = nn.Sequential()
        self.local_rep.add_module('conv_3x3', conv_2d(inp, inp, kernel_size=3, stride=1, padding=1, groups=inp))
        self.local_rep.add_module('conv_1x1', conv_2d(inp, attn_dim, kernel_size=1, stride=1, norm=False, act=False))

        # global representation
        self.global_rep = nn.Sequential()
        ffn_dims = [int((ffn_multiplier * attn_dim) // 16 * 16)] * attn_blocks
        for i in range(attn_blocks):
            ffn_dim = ffn_dims[i]
            self.global_rep.add_module(f'TransformerEncoder_{i}',
                                       TransformerEncoder(attn_dim, ffn_dim, heads, dim_head))
        self.global_rep.add_module('LayerNorm', nn.LayerNorm(attn_dim, eps=1e-5, elementwise_affine=True))

        self.conv_proj = conv_2d(attn_dim, inp, kernel_size=1, stride=1)
        self.fusion = conv_2d(inp + attn_dim, inp, kernel_size=1, stride=1)

    def unfolding(self, feature_map):
        patch_w, patch_h = self.patch_w, self.patch_h
        batch_size, in_channels, orig_h, orig_w = feature_map.shape

        new_h = int(math.ceil(orig_h / self.patch_h) * self.patch_h)
        new_w = int(math.ceil(orig_w / self.patch_w) * self.patch_w)

        interpolate = False
        if new_w != orig_w or new_h != orig_h:
            # Note: Padding can be done, but then it needs to be handled in attention function.
            feature_map = F.interpolate(
                feature_map, size=(new_h, new_w), mode="bilinear", align_corners=False
            )
            interpolate = True

        # number of patches along width and height
        num_patch_w = new_w // patch_w  # n_w
        num_patch_h = new_h // patch_h  # n_h
        num_patches = num_patch_h * num_patch_w  # N

        # [B, C, H, W] --> [B * C * n_h, p_h, n_w, p_w]
        reshaped_fm = feature_map.reshape(
            batch_size * in_channels * num_patch_h, patch_h, num_patch_w, patch_w
        )
        # [B * C * n_h, p_h, n_w, p_w] --> [B * C * n_h, n_w, p_h, p_w]
        transposed_fm = reshaped_fm.transpose(1, 2)
        # [B * C * n_h, n_w, p_h, p_w] --> [B, C, N, P] where P = p_h * p_w and N = n_h * n_w
        reshaped_fm = transposed_fm.reshape(
            batch_size, in_channels, num_patches, self.patch_area
        )
        # [B, C, N, P] --> [B, P, N, C]
        transposed_fm = reshaped_fm.transpose(1, 3)
        # [B, P, N, C] --> [BP, N, C]
        patches = transposed_fm.reshape(batch_size * self.patch_area, num_patches, -1)

        info_dict = {
            "orig_size": (orig_h, orig_w),
            "batch_size": batch_size,
            "interpolate": interpolate,
            "total_patches": num_patches,
            "num_patches_w": num_patch_w,
            "num_patches_h": num_patch_h,
        }

        return patches, info_dict

    def folding(self, patches, info_dict):
        n_dim = patches.dim()
        assert n_dim == 3, "Tensor should be of shape BPxNxC. Got: {}".format(
            patches.shape
        )
        # [BP, N, C] --> [B, P, N, C]
        patches = patches.contiguous().view(
            info_dict["batch_size"], self.patch_area, info_dict["total_patches"], -1
        )

        batch_size, pixels, num_patches, channels = patches.size()
        num_patch_h = info_dict["num_patches_h"]
        num_patch_w = info_dict["num_patches_w"]

        # [B, P, N, C] --> [B, C, N, P]
        patches = patches.transpose(1, 3)

        # [B, C, N, P] --> [B*C*n_h, n_w, p_h, p_w]
        feature_map = patches.reshape(
            batch_size * channels * num_patch_h, num_patch_w, self.patch_h, self.patch_w
        )
        # [B*C*n_h, n_w, p_h, p_w] --> [B*C*n_h, p_h, n_w, p_w]
        feature_map = feature_map.transpose(1, 2)
        # [B*C*n_h, p_h, n_w, p_w] --> [B, C, H, W]
        feature_map = feature_map.reshape(
            batch_size, channels, num_patch_h * self.patch_h, num_patch_w * self.patch_w
        )
        if info_dict["interpolate"]:
            feature_map = F.interpolate(
                feature_map,
                size=info_dict["orig_size"],
                mode="bilinear",
                align_corners=False,
            )
        return feature_map

    def forward(self, x):
        res = x.clone()
        fm_conv = self.local_rep(x)
        x, info_dict = self.unfolding(fm_conv)
        x = self.global_rep(x)
        x = self.folding(x, info_dict)
        x = self.conv_proj(x)
        x = self.fusion(torch.cat((fm_conv, x), dim=1))
        x = x + res
        return x


class MobileViTv3(nn.Module):
    def __init__(self, image_size=(192, 192), mode='small', patch_size=(2, 2), c=3):
        """
        Implementation of MobileViTv3 based on v1
        """
        super().__init__()
        # check image size
        ih, iw = image_size
        self.ph, self.pw = patch_size
        assert ih % self.ph == 0 and iw % self.pw == 0
        assert mode in ['xx_small', 'x_small', 'small']

        # model size
        if mode == 'xx_small':
            mv2_exp_mult = 2
            ffn_multiplier = 2
            last_layer_exp_factor = 4
            channels = [16, 16, 24, 64, 80, 128]
            attn_dim = [64, 80, 96]
        elif mode == 'x_small':
            mv2_exp_mult = 4
            ffn_multiplier = 2
            last_layer_exp_factor = 4
            channels = [16, 32, 48, 96, 160, 160]
            attn_dim = [96, 120, 144]
        elif mode == 'small':
            mv2_exp_mult = 4
            ffn_multiplier = 2
            last_layer_exp_factor = 3
            channels = [16, 32, 64, 128, 256, 320]
            attn_dim = [144, 192, 240]
        else:
            raise NotImplementedError

        self.conv_0 = conv_2d(c, channels[0], kernel_size=3, stride=1, padding=1)

        self.layer_1 = nn.Sequential(
            InvertedResidual(channels[0], channels[1], stride=1, expand_ratio=mv2_exp_mult)
        )
        self.layer_2 = nn.Sequential(
            InvertedResidual(channels[1], channels[2], stride=2, expand_ratio=mv2_exp_mult),
            InvertedResidual(channels[2], channels[2], stride=1, expand_ratio=mv2_exp_mult),
            InvertedResidual(channels[2], channels[2], stride=1, expand_ratio=mv2_exp_mult)
        )
        self.layer_3 = nn.Sequential(
            InvertedResidual(channels[2], channels[3], stride=2, expand_ratio=mv2_exp_mult),
            MobileViTBlockV3(channels[3], attn_dim[0], ffn_multiplier, heads=4, dim_head=8, attn_blocks=2,
                                patch_size=patch_size)
        )
        self.layer_4 = nn.Sequential(
            InvertedResidual(channels[3], channels[4], stride=2, expand_ratio=mv2_exp_mult),
            MobileViTBlockV3(channels[4], attn_dim[1], ffn_multiplier, heads=4, dim_head=8, attn_blocks=4,
                                patch_size=patch_size)
        )
        self.layer_5 = nn.Sequential(
            InvertedResidual(channels[4], channels[5], stride=2, expand_ratio=mv2_exp_mult),
            MobileViTBlockV3(channels[5], attn_dim[2], ffn_multiplier, heads=4, dim_head=8, attn_blocks=3,
                                patch_size=patch_size)
        )

    def forward(self, x):
        x = self.conv_0(x)
        x1 = self.layer_1(x)
        x2 = self.layer_2(x1)
        x3 = self.layer_3(x2)
        x4 = self.layer_4(x3)
        x5 = self.layer_5(x4)

        return [x1, x2, x3, x4, x5]

######Contextual Self-Correlation Semantic Unification Module######


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class CSSU(nn.Module):
    def __init__(self, in_channels, ou_channels, h, w):
        super(CSSU, self).__init__()
        self.conv1 = nn.Conv2d(ou_channels, in_channels, 3, 1, 1)
        self.conv2 = BasicConv2d(2 * in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.conv_q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv_k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv_v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.delta = nn.Parameter(torch.Tensor(1, in_channels, h, w).float(), requires_grad=True)
        nn.init.normal_(self.delta, mean=0, std=0.01)
        self.avg_pool = nn.AvgPool2d((3, 3), stride=1, padding=1)
        self.amg_pool = nn.MaxPool2d((3, 3), stride=1, padding=1)
        self.conv_rbc = BasicConv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.conv_rbc_2 = BasicConv2d(2 * in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.rl = nn.ReLU(inplace=True)
        self.ca = Channelattention(in_channels)
        self.conv_rbc_3 = BasicConv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
    def forward(self, r, d):
        B, C, H, W = r.size()
        y = self.conv1(d)
        y = F.interpolate(y, (H, W), mode='bilinear', align_corners=True)
        r = self.conv2(torch.cat([r, y], dim=1))
        P = H * W
        ftr_q = self.conv_q(r).view(B, -1, P).permute(0, 2, 1)
        ftr_k = self.conv_k(r).view(B, -1, P)
        ftr_v = self.conv_v(r).view(B, -1, P)
        weights = F.softmax(torch.bmm(ftr_q, ftr_k), dim=1)
        G = torch.bmm(ftr_v, weights).view(B, C, H, W)
        out = self.delta * G + r
        out =torch.sigmoid(self.conv_rbc(out))
        ftr_avg = self.avg_pool(r)
        ftr_max = self.amg_pool(r)
        ftr_avg =ftr_avg.mul(out)
        ftr_max =ftr_max.mul(out)
        f = self.conv3(ftr_avg + ftr_max)
        f = self.bn(f)
        f_rl = self.rl(f)
        out =torch.cat((f_rl, r), 1)
        out = self.conv_rbc_2(out)
        out = self.ca(out) * out
        out = self.conv_rbc_3(out)
        return out

class Channelattention(nn.Module):   #CA
    def __init__(self, in_planes):
        super(Channelattention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 4, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 4, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return self.sigmoid(max_out)


class Spatialattention(nn.Module):  #SA
    def __init__(self, kernel_size=7):
        super(Spatialattention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)


######Cross-Scale Spatial Feature Refinement Module######


class GCN(nn.Module):
    def __init__(self, num_state, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1, padding=0,
                               stride=1, groups=1, bias=True)
        self.relu = nn.LeakyReLU(0.2,inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, padding=0,
                               stride=1, groups=1, bias=bias)

    def forward(self, x):
        h = self.conv1(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1)
        h = h + x
        h = self.relu(h)
        h = self.conv2(h)
        return h

class FI(nn.Module):
    def __init__(self, num_in, num_mid, kernel=1):
        super(FI, self).__init__()
        self.num_s = int(2 * num_mid)
        self.num_n = int(1 * num_mid)
        kernel_size = (kernel, kernel)
        padding = (1, 1) if kernel == 3 else (0, 0)
        self.conv_state = nn.Sequential(nn.Conv2d(num_in, self.num_s,  kernel_size=kernel_size, padding=padding), nn.BatchNorm2d(self.num_s))
        self.conv_proj = nn.Sequential(nn.Conv2d(num_in, self.num_n,  kernel_size=kernel_size, padding=padding), nn.BatchNorm2d(self.num_n))
        self.conv_state2 = nn.Sequential(nn.Conv2d(num_in, self.num_s,  kernel_size=kernel_size, padding=padding), nn.BatchNorm2d(self.num_s))
        self.gcn1 = GCN(num_state=self.num_s, num_node=self.num_n)
        self.gcn2 = GCN(num_state=self.num_s, num_node=self.num_n)
        self.fc_2 = nn.Conv2d(num_in, num_in, kernel_size=kernel_size, padding=padding, stride=(1, 1), groups=1, bias=False)
        self.blocker = nn.BatchNorm2d(num_in)

    def forward(self, x, y):
        batch_size = x.size(0)
        x_state_reshaped = self.conv_state(x).view(batch_size, self.num_s, -1)
        y_proj_reshaped = self.conv_proj(y).view(batch_size, self.num_n, -1)
        x_state_2 = self.conv_state2(x).view(batch_size, self.num_s, -1)
        x_n_state1 = torch.bmm(x_state_reshaped, y_proj_reshaped.permute(0, 2, 1))
        x_n_state2 = x_n_state1 * (1. / x_state_reshaped.size(2))
        x_n_rel1 = self.gcn1(x_n_state2)
        x_n_rel2 = self.gcn2(x_n_rel1)
        x_state_reshaped = torch.bmm(x_n_rel2.permute(0,2,1), x_state_2)
        B, C, N = x_state_reshaped.shape
        x_state = x_state_reshaped.view(batch_size, 1, int(sqrt(N)), int(sqrt(N)))
        out = x + self.blocker(self.fc_2(x_state))
        return out


class convbnrelu(nn.Module):

    def __init__(self, in_channel, out_channel, k=3, s=1, p=1, g=1, d=1, bias=False, bn=True, relu=True):
        super(convbnrelu, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
        if relu:
            conv.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)

class OLDC(nn.Module):
    def __init__(self, channel):
        super(OLDC, self).__init__()
        self.channel = channel
        self.fconv = nn.ModuleList([nn.Sequential(
            nn.Conv2d(channel, channel, (3, 1), padding=(1, 0), groups=channel, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)), nn.Sequential(
            nn.Conv2d(channel, channel, (1, 3), padding=(0, 1), groups=channel, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)), nn.Sequential(
            nn.Conv2d(channel, channel, (3, 3), padding=(1, 1), groups=channel, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True))])
        self.Dconv = nn.ModuleList([convbnrelu(channel, channel, k=3, s=1, p=i, d=i, g=channel) for i in [1, 2, 3, 4]])

        self.Pconv = nn.ModuleList([convbnrelu(3, 1, k=1, s=1, p=0) for _ in range(channel)])

        self.Pfusion = convbnrelu(channel, channel, k=1, s=1, p=0)

        self.relu = nn.ReLU(inplace=True)

    def shuffle(self, F, channel):
        Ms = [torch.chunk(x, channel, dim=1) for x in F]
        I = [torch.cat((Ms[0][i], Ms[1][i], Ms[2][i]), 1) for i in range(channel)]
        return I

    def forward(self, x):
        M = [conv(x) for conv in self.fconv]
        IF = self.shuffle(M, self.channel)
        IF = [self.Pconv[i](IF[i]) for i in range(self.channel)]
        IF = [torch.squeeze(x, dim=1) for x in IF]
        out = self.Pfusion(torch.stack(IF, 1))
        out = self.relu(out + x)
        return out

class CSFR(nn.Module):
    def __init__(self, in_channel, ou_channel):
        super(CSFR, self).__init__()
        self.ca1 = Channelattention(in_channel)
        self.ca2 = Channelattention(in_channel)
        self.sa1 = Spatialattention()
        self.sa2 = Spatialattention()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=2 * in_channel, out_channels=1, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(1),
                                   nn.Sigmoid())
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=in_channel, out_channels=1, kernel_size=3, padding=1, dilation=1),
                                   nn.BatchNorm2d(1),
                                   nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=in_channel, out_channels=1, kernel_size=3, padding=1, dilation=1),
                                   nn.BatchNorm2d(1),
                                   nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels=in_channel+1, out_channels=in_channel, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(in_channel),
                                   nn.ReLU())
        self.conv5 = nn.Conv2d(ou_channel, in_channel, 3, 1, 1)
        self.ghl1 = FI(1, 1)
        self.ghl2 = FI(1, 1)
        self.mir = OLDC(in_channel)
        self.mid = OLDC(in_channel)
        self.delta = nn.Parameter(torch.Tensor([0.1]))
    def forward(self, r, d):
        xsize = r.size()[2:]
        d = self.conv5(d)
        d = F.interpolate(d, xsize, mode='bilinear', align_corners=True)
        rx = self.ca1(r) * r
        dx = self.ca2(d) * d
        rdx = torch.cat((rx,dx),dim=1)
        rdx = self.conv1(rdx)
        rxg = self.conv2(rx)
        dxg = self.conv3(dx)
        rxg = self.ghl1(rxg, dxg)
        dxg = self.ghl2(dxg, rxg)
        c1 = rdx.mul(rxg)
        c2 = rdx.mul(dxg)
        rx = self.mir(rx)
        dx = self.mid(dx)
        dxa = self.sa1(rx)*dx+dx
        rxa = self.sa2(dx)*rx+rx
        c1 = torch.cat((c1,rxa),dim=1)
        c2 = torch.cat((c2,dxa),dim=1)
        res1 = c1+self.delta*c2
        res2 = self.conv4(res1)
        out = res2+r+d
        return out

class CDARNet(nn.Module):
    def __init__(self, inchannel=3, nclass=4):
        super(CDARNet, self).__init__()
        self.partnet = MobileViTv3(c=inchannel, mode='xx_small')
        self.rd4 = CSSU(80, 128, 24, 24)
        self.rd3 = CSSU(64, 80, 48, 48)
        self.rd2 = CSFR(24, 64)
        self.rd1 = CSFR(16, 24)
        self.conv_1x1 = nn.Conv2d(16, nclass, 1)
        self.unfold = nn.Unfold((3, 3), stride=3)
        self.fold1 = nn.Fold(kernel_size=(3, 3), stride=3, output_size=(192, 192))
        self.fold2 = nn.Fold(kernel_size=(3, 3), stride=3, output_size=(96, 96))
        self.fold3 = nn.Fold(kernel_size=(3, 3), stride=3, output_size=(48, 48))
        self.fold4 = nn.Fold(kernel_size=(3, 3), stride=3, output_size=(24, 24))
        self.fold5 = nn.Fold(kernel_size=(3, 3), stride=3, output_size=(12, 12))
        self.line = nn.Linear(27, 3)
        self.line_re1 = nn.Linear(16, 144)
        self.line_re2 = nn.Linear(24, 216)
        self.line_re3 = nn.Linear(64, 576)
        self.line_re4 = nn.Linear(80, 720)
        self.line_re5 = nn.Linear(128, 1152)
        self.apply(self._init_weights)

    def forward(self, input):
        b, c, h, w = input.size()
        region_input = self.unfold(input).permute(0, 2, 1).contiguous()
        region_input = self.line(region_input).permute(0, 2, 1).contiguous().reshape(b, c, 64, 64)
        p_f = self.partnet(region_input)
        out1 = self.fold1(self.line_re1(p_f[0].reshape(b, 16, -1).permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous())
        out2 = self.fold2(self.line_re2(p_f[1].reshape(b, 24, -1).permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous())
        out3 = self.fold3(self.line_re3(p_f[2].reshape(b, 64, -1).permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous())
        out4 = self.fold4(self.line_re4(p_f[3].reshape(b, 80, -1).permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous())
        out5 = self.fold5(self.line_re5(p_f[4].reshape(b, 128, -1).permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous())
        F5 = out5
        F4 = out4
        F3 = out3
        F2 = out2
        F1 = out1
        F4_ = self.rd4(F4, F5)
        F3_ = self.rd3(F3, F4_)
        F2_ = self.rd2(F2, F3_)
        F1_ = self.rd1(F1, F2_)
        out = self.conv_1x1(F1_)
        return out

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.01)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()