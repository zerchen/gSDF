#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
#@File        :transformer.py
#@Date        :2022/09/29 15:06:45
#@Author      :zerui chen
#@Contact     :zerui.chen@inria.fr

import torch
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class VideoTransformer(nn.Module):
    def __init__(self, image_size, num_frames, dim=256, depth=4, heads=4, in_channels=256, dim_head=64, dropout=0., scale_dim=4, use_temporary_embedding=False, patch_size=1, sep_output=True):
        super().__init__()
        self.sep_output = sep_output
        self.patch_size = patch_size
        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.image_size = image_size
        if use_temporary_embedding:
            self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, num_patches, dim))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(1, 1, num_patches, dim))
        self.spatial_transformer = Transformer(dim, depth, heads, dim_head, dim * scale_dim, dropout)
        self.temporal_transformer = Transformer(dim, depth, heads, dim_head, dim * scale_dim, dropout)

    def forward(self, x):
        x = self.to_patch_embedding(x)
        b, t, n, _ = x.shape

        x += self.pos_embedding[:, :t, :n]
        x = rearrange(x, 'b t n d -> (b t) n d')
        x = self.spatial_transformer(x)
        x = rearrange(x, '(b t) n d -> (b n) t d', t=t)
        x = self.temporal_transformer(x)
        x = rearrange(x, '(b n) t d -> b t d n', n=n)
        x = rearrange(x, 'b t d (h w) -> b t d h w', h=self.image_size // self.patch_size, w=self.image_size // self.patch_size)

        if self.sep_output:
            feat_list = []
            for i in range(t):
                feat_list.append(x[:, i, :, :, :])
            return feat_list
        else:
            return x


class FSATransformer(nn.Module):
    """Factorized Self-Attention Transformer Encoder"""
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList(
                [PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                 PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                 PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
                 ]))

    def forward(self, x):
        b, t, n, _ = x.shape
        x = rearrange(x, 'b t n d -> (b t) n d')

        for sp_attn, temp_attn, ff in self.layers:
            sp_attn_x = sp_attn(x) + x  # Spatial attention

            # Reshape tensors for temporal attention
            sp_attn_x = rearrange(sp_attn_x, '(b t) n d -> (b n) t d', n=n, t=t)
            temp_attn_x = temp_attn(sp_attn_x) + sp_attn_x  # Temporal attention
            x = ff(temp_attn_x) + temp_attn_x  # MLP

            # Again reshape tensor for spatial attention
            x = rearrange(x, '(b n) t d -> (b t) n d', n=n, t=t)

        x = rearrange(x, '(b t) n d -> b t d n', t=t, n=n)

        return x


class FactorizedVideoTransformer(nn.Module):
    def __init__(self, image_size, num_frames, dim=256, depth=4, heads=4, in_channels=256, dim_head=64, dropout=0., scale_dim=4, use_temporary_embedding=False, patch_size=1, sep_output=True):
        super().__init__()
        self.sep_output = sep_output
        self.patch_size = patch_size
        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.image_size = image_size
        if use_temporary_embedding:
            self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, num_patches, dim))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(1, 1, num_patches, dim))
        self.transformer = FSATransformer(dim, depth, heads, dim_head, dim * scale_dim, dropout)

    def forward(self, x):
        x = self.to_patch_embedding(x)
        b, t, n, _ = x.shape

        x += self.pos_embedding[:, :t, :n]
        x = self.transformer(x)
        x = rearrange(x, 'b t d (h w) -> b t d h w', h=self.image_size // self.patch_size, w=self.image_size // self.patch_size)

        if self.sep_output:
            feat_list = []
            for i in range(t):
                feat_list.append(x[:, i, :, :, :])
            return feat_list
        else:
            return x