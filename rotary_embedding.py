# Copyright (c) 2023, Tri Dao.

import math
from typing import Optional, Tuple
import time

import rotary_emb
from rotary_embedding_triton import apply_rotary
import torch
from einops import rearrange, repeat

class ApplyRotaryEmb(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cos, sin, interleaved=False, inplace=False):
        """
            x: (batch_size, seqlen, nheads, headdim)
            cos, sin: (seqlen, rotary_dim / 2)
            interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead
                of 1st half and 2nd half (GPT-NeoX style).
        rotary_dim must be <= headdim
        Apply rotary embedding to the first rotary_dim of x.
        """
        batch, seqlen, nheads, headdim = x.shape
        rotary_seqlen, rotary_dim = cos.shape
        rotary_dim *= 2
        assert rotary_dim <= headdim
        assert seqlen <= rotary_seqlen
        assert sin.shape == (rotary_seqlen, rotary_dim // 2)
        x_ro = x[..., :rotary_dim]
        x1, x2 = x_ro.chunk(2, dim=-1) if not interleaved else (x_ro[..., ::2], x_ro[..., 1::2])
        # print('x1 dtype:', x1.dtype)
        # print('cos dtype:', cos.dtype)
        # print('///////')
        # print('x1\n', x1)
        # print('===========')
        # print('x2\n', x2)
        out = torch.empty_like(x) if not inplace else x
        out_ro = out[..., :rotary_dim]
        if inplace:
            o1, o2 = x1, x2
        else:
            o1, o2 = (
                out_ro.chunk(2, dim=-1)
                if not interleaved
                else (out_ro[..., ::2], out_ro[..., 1::2])
            )
        # print('x.shape:', x.shape)
        # print('x1.shape:', x1.shape)
        # print('x2.shape:', x2.shape)
        # print('cos shape:', cos[:seqlen].shape)
        # print('cos[:seqlen].shape:', cos[:seqlen].shape)
        # print('sin[:seqlen].shape:', sin[:seqlen].shape)
        # print('o1.shape:', o1.shape)
        # print('o2.shape:', o2.shape)
        rotary_emb.apply_rotary(
            x1,
            x2,
            rearrange(cos[:seqlen], "s d -> s 1 d"),
            rearrange(sin[:seqlen], "s d -> s 1 d"),
            o1,
            o2,
            False,
        )
        # rotary_emb.apply_rotary(
        #     x1,
        #     x2,
        #     cos[:seqlen],
        #     sin[:seqlen],
        #     o1,
        #     o2,
        #     False,
        # )
        # print('o1\n', o1)
        # print('o2\n', o2)
        if not inplace and rotary_dim < headdim:
            out[..., rotary_dim:].copy_(x[..., rotary_dim:])
        ctx.save_for_backward(cos, sin)
        ctx.interleaved = interleaved
        ctx.inplace = inplace
        return out if not inplace else x

    @staticmethod
    def backward(ctx, do):
        cos, sin = ctx.saved_tensors
        _, seqlen, _, headdim = do.shape
        rotary_dim = cos.shape[-1]
        rotary_dim *= 2
        inplace = ctx.inplace
        do_ro = do[..., :rotary_dim]
        do1, do2 = (
            do_ro.chunk(2, dim=-1) if not ctx.interleaved else (do_ro[..., ::2], do_ro[..., 1::2])
        )
        dx = torch.empty_like(do) if not inplace else do
        if inplace:
            dx1, dx2 = do1, do2
        else:
            dx_ro = dx[..., :rotary_dim]
            dx1, dx2 = (
                dx_ro.chunk(2, dim=-1)
                if not ctx.interleaved
                else (dx_ro[..., ::2], dx_ro[..., 1::2])
            )
        rotary_emb.apply_rotary(
            do1,
            do2,
            rearrange(cos[:seqlen], "s d -> s 1 d"),
            rearrange(sin[:seqlen], "s d -> s 1 d"),
            dx1,
            dx2,
            True,
        )
        if not inplace and rotary_dim < headdim:
            dx[..., rotary_dim:].copy_(do[..., rotary_dim:])
        return dx, None, None, None, None
    
    
class ApplyRotaryEmb2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cos, sin, interleaved=False, inplace=False):
        """
            x: (batch_size, seqlen, nheads, headdim)
            cos, sin: (seqlen, rotary_dim / 2)
            interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead
                of 1st half and 2nd half (GPT-NeoX style).
        rotary_dim must be <= headdim
        Apply rotary embedding to the first rotary_dim of x.
        """
        batch, seqlen, nheads, headdim = x.shape
        rotary_seqlen, rotary_dim = cos.shape
        rotary_dim *= 2
        assert rotary_dim <= headdim
        assert seqlen <= rotary_seqlen
        assert sin.shape == (rotary_seqlen, rotary_dim // 2)
        x = x[..., :rotary_dim]
        out = torch.empty_like(x) if not inplace else x
        # print('x.shape:', x.shape)
        # print('x1.shape:', x1.shape)
        # print('x2.shape:', x2.shape)
        # print('cos shape:', cos[:seqlen].shape)
        # print('cos[:seqlen].shape:', cos[:seqlen].shape)
        # print('sin[:seqlen].shape:', sin[:seqlen].shape)
        # print('o1.shape:', o1.shape)
        # print('o2.shape:', o2.shape)
        rotary_emb.apply_rotary2(
            x,
            cos,
            sin,
            out,
            False,
        )
        # print('o1\n', out)
        if not inplace and rotary_dim < headdim:
            out[..., rotary_dim:].copy_(x[..., rotary_dim:])
        ctx.save_for_backward(cos, sin)
        ctx.interleaved = interleaved
        ctx.inplace = inplace
        return out if not inplace else x

    @staticmethod
    def backward(ctx, do):
        cos, sin = ctx.saved_tensors
        _, seqlen, _, headdim = do.shape
        rotary_dim = cos.shape[-1]
        rotary_dim *= 2
        inplace = ctx.inplace
        do_ro = do[..., :rotary_dim]
        dx = torch.empty_like(do) if not inplace else do
        rotary_emb.apply_rotary2(
            do,
            cos,
            sin,
            dx,
            True,
        )
        if not inplace and rotary_dim < headdim:
            dx[..., rotary_dim:].copy_(do[..., rotary_dim:])
        return dx, None, None, None, None
    
class ApplyRotaryEmb3(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cos, sin, interleaved=False, inplace=False):
        """
            x: (batch_size, seqlen, nheads, headdim)
            cos, sin: (seqlen, rotary_dim / 2)
            interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead
                of 1st half and 2nd half (GPT-NeoX style).
        rotary_dim must be <= headdim
        Apply rotary embedding to the first rotary_dim of x.
        """
        batch, seqlen, nheads, headdim = x.shape
        rotary_seqlen, rotary_dim = cos.shape
        rotary_dim *= 2
        assert rotary_dim <= headdim
        assert seqlen <= rotary_seqlen
        assert sin.shape == (rotary_seqlen, rotary_dim // 2)
        x = x[..., :rotary_dim]
        out = torch.empty_like(x) if not inplace else x
        # print('x.shape:', x.shape)
        # print('x1.shape:', x1.shape)
        # print('x2.shape:', x2.shape)
        # print('cos shape:', cos[:seqlen].shape)
        # print('cos[:seqlen].shape:', cos[:seqlen].shape)
        # print('sin[:seqlen].shape:', sin[:seqlen].shape)
        # print('o1.shape:', o1.shape)
        # print('o2.shape:', o2.shape)
        rotary_emb.apply_rotary3(
            x,
            cos,
            sin,
            out,
            False,
        )
        # print('o1\n', out)
        if not inplace and rotary_dim < headdim:
            out[..., rotary_dim:].copy_(x[..., rotary_dim:])
        ctx.save_for_backward(cos, sin)
        ctx.interleaved = interleaved
        ctx.inplace = inplace
        return out if not inplace else x

    @staticmethod
    def backward(ctx, do):
        cos, sin = ctx.saved_tensors
        _, seqlen, _, headdim = do.shape
        rotary_dim = cos.shape[-1]
        rotary_dim *= 2
        inplace = ctx.inplace
        do_ro = do[..., :rotary_dim]
        dx = torch.empty_like(do) if not inplace else do
        rotary_emb.apply_rotary3(
            do,
            cos,
            sin,
            dx,
            True,
        )
        if not inplace and rotary_dim < headdim:
            dx[..., rotary_dim:].copy_(do[..., rotary_dim:])
        return dx, None, None, None, None


class ApplyRotaryEmbTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cos, sin, interleaved=False, inplace=False):
        """
            x: (batch_size, seqlen, nheads, headdim)
            cos, sin: (seqlen, rotary_dim / 2)
            interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead
                of 1st half and 2nd half (GPT-NeoX style).
        rotary_dim must be <= headdim
        Apply rotary embedding to the first rotary_dim of x.
        """
        batch, seqlen, nheads, headdim = x.shape
        rotary_seqlen, rotary_dim = cos.shape
        rotary_dim *= 2
        assert rotary_dim <= headdim
        assert seqlen <= rotary_seqlen
        assert sin.shape == (rotary_seqlen, rotary_dim // 2)
        x = x[..., :rotary_dim]
        out = torch.empty_like(x) if not inplace else x
        # print('x.shape:', x.shape)
        # print('x1.shape:', x1.shape)
        # print('x2.shape:', x2.shape)
        # print('cos shape:', cos[:seqlen].shape)
        # print('cos[:seqlen].shape:', cos[:seqlen].shape)
        # print('sin[:seqlen].shape:', sin[:seqlen].shape)
        # print('o1.shape:', o1.shape)
        # print('o2.shape:', o2.shape)
        apply_rotary(
            x,
            cos,
            sin,
            out,
            False,
        )
        # print('o1\n', out)
        if not inplace and rotary_dim < headdim:
            out[..., rotary_dim:].copy_(x[..., rotary_dim:])
        ctx.save_for_backward(cos, sin)
        ctx.interleaved = interleaved
        ctx.inplace = inplace
        return out if not inplace else x

    @staticmethod
    def backward(ctx, do):
        cos, sin = ctx.saved_tensors
        _, seqlen, _, headdim = do.shape
        rotary_dim = cos.shape[-1]
        rotary_dim *= 2
        inplace = ctx.inplace
        do_ro = do[..., :rotary_dim]
        dx = torch.empty_like(do) if not inplace else do
        apply_rotary(
            do,
            cos,
            sin,
            dx,
            True,
        )
        if not inplace and rotary_dim < headdim:
            dx[..., rotary_dim:].copy_(do[..., rotary_dim:])
        return dx, None, None, None, None



apply_rotary_emb_func = ApplyRotaryEmb.apply
apply_rotary_emb_func2 = ApplyRotaryEmb2.apply
apply_rotary_emb_func3 = ApplyRotaryEmb3.apply
apply_rotary_emb_triton = ApplyRotaryEmbTriton.apply

# def apply_rotary_emb_func(x, cos, sin, interleaved=False, inplace=False):
#         """
#             x: (batch_size, seqlen, nheads, headdim)
#             cos, sin: (seqlen, rotary_dim / 2)
#             interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead
#                 of 1st half and 2nd half (GPT-NeoX style).
#         rotary_dim must be <= headdim
#         Apply rotary embedding to the first rotary_dim of x.
#         """
#         batch, seqlen, nheads, headdim = x.shape
#         rotary_seqlen, rotary_dim = cos.shape
#         rotary_dim *= 2
#         assert rotary_dim <= headdim
#         assert seqlen <= rotary_seqlen
#         assert sin.shape == (rotary_seqlen, rotary_dim // 2)
#         # x_ro = x[..., :rotary_dim]
#         x_ro = x
#         # torch.cuda.synchronize()
#         # t1 = time.time()
#         x1, x2 = x_ro.chunk(2, dim=-1) if not interleaved else (x_ro[..., ::2], x_ro[..., 1::2])
#         t2 = time.time()
#         # torch.cuda.synchronize()
#         # print('time taken for chunking:', (t2 - t1) * 1e9, 'ns')
#         # print('///////')
#         # print(x1)
#         # print('===========')
#         # print(x2)
#         # torch.cuda.synchronize()
#         # t1 = time.time()
#         out = torch.empty_like(x) if not inplace else x
#         # out_ro = out[..., :rotary_dim]
#         out_ro = out
#         if inplace:
#             o1, o2 = x1, x2
#         else:
#             o1, o2 = (
#                 out_ro.chunk(2, dim=-1)
#                 if not interleaved
#                 else (out_ro[..., ::2], out_ro[..., 1::2])
#             )
#         # t2 = time.time()
#         # torch.cuda.synchronize()
#         # print('time taken for output:', (t2 - t1) * 1e9, 'ns')
#         # print('x.shape:', x.shape)
#         # print('x1.shape:', x1.shape)
#         # print('x2.shape:', x2.shape)
#         # print('cos[:seqlen].shape:', cos[:seqlen].shape)
#         # print('sin[:seqlen].shape:', sin[:seqlen].shape)
#         # print('o1.shape:', o1.shape)
#         # print('o2.shape:', o2.shape)
#         # torch.cuda.synchronize()
#         # t1 = time.time()
#         rotary_emb.apply_rotary(
#             x1,
#             x2,
#             rearrange(cos[:seqlen], "s d -> s 1 d"),
#             rearrange(sin[:seqlen], "s d -> s 1 d"),
#             o1,
#             o2,
#             False,
#         )
#         # t2 = time.time()
#         # torch.cuda.synchronize()
#         # print('time taken for function:', (t2 - t1) * 1e9, 'ns')
#         # torch.cuda.synchronize()
#         # t1 = time.time()
#         if not inplace and rotary_dim < headdim:
#             print('not in place')
#             out[..., rotary_dim:].copy_(x[..., rotary_dim:])
#         # ctx.save_for_backward(cos, sin)
#         # ctx.interleaved = interleaved
#         # ctx.inplace = inplace
#         # t2 = time.time()
#         # torch.cuda.synchronize()
#         # print('time taken for output and context:', (t2 - t1) * 1e9, 'ns')
#         return out if not inplace else x