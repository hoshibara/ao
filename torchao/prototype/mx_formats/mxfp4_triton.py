# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Hand-written (pure Triton, no inline PTX) mxfp4 dim0 cast kernel.

This module demonstrates how to:
  1. Implement quantize (bf16/fp16/fp32 -> fp4 e2m1) + pack (2 nibbles per byte)
     entirely inside a single Triton kernel using only ``tl`` ops (no PTX
     ``cvt.e2m1x2`` instruction), so it runs on any Triton-capable GPU.
  2. Register that kernel as a PyTorch custom op via ``torch.library.triton_op``
     + ``wrap_triton`` so that ``torch.compile`` can take it over as a single
     opaque graph node, instead of Inductor lowering the equivalent eager code
     (amax -> e8m0 scale -> quantize -> ``pack_uint4``) into >= 2 kernels.

The output convention matches ``torchao.prototype.mx_formats.mx_tensor.to_mx``
for ``elem_dtype == torch.float4_e2m1fn_x2`` with ``ScaleCalculationMode.FLOOR``:
  * ``data``:  uint8 viewed as ``float4_e2m1fn_x2``, last dim packed (n_cols // 2)
  * ``scale``: uint8 viewed as ``float8_e8m0fnu``, shape (n_rows, n_cols // block)

NOTE: this is prototype / reference code intended for validation. The
round-to-nearest-even / subnormal handling for f32 -> e2m1 is implemented with
plain bit ops and matches the OCP MX e2m1 encoding, but has not been tuned for
performance.
"""

from typing import Tuple

import torch
from torch.utils._triton import has_triton

__all__ = [
    "triton_to_mxfp4_dim0",
    "mxfp4_available",
]


def mxfp4_available() -> bool:
    return has_triton()


if has_triton():
    import triton
    import triton.language as tl
    from torch.library import triton_op, wrap_triton

    # e8m0 / e2m1 / fp32 constants
    EBITS_F4_E2M1: tl.constexpr = 2
    MBITS_F4_E2M1: tl.constexpr = 1
    F4_E2M1_MAX: tl.constexpr = 6.0
    E8M0_EXP_BIAS: tl.constexpr = 127
    F32_MBITS: tl.constexpr = 23
    F32_EBITS: tl.constexpr = 8
    F32_EXP_BIAS: tl.constexpr = 127

    @triton.jit
    def _f32_to_e2m1_unsigned_code(x_abs):
        """Map a non-negative fp32 value to its 3-bit e2m1 magnitude code (0..7).

        e2m1 representable magnitudes: {0, 0.5, 1, 1.5, 2, 3, 4, 6}
        with codes                    {0,   1, 2,   3, 4, 5, 6, 7}.

        Uses a round-to-nearest-even comparison ladder on the midpoints
        {0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0}. Ties go to even code.
        """
        # Saturate inf / nan / >max to the max magnitude (code 7 == 6.0).
        is_bad = (x_abs != x_abs) | (x_abs > 3.0e38)
        x_abs = tl.where(is_bad, F4_E2M1_MAX, x_abs)

        # Start at code 0 (== 0.0) and climb the ladder.
        code = tl.zeros(x_abs.shape, dtype=tl.int32)
        # 0.25: even tie -> stays 0
        code = tl.where(x_abs > 0.25, 1, code)
        # 0.75: odd tie -> rounds up to 1.0 (code 2)
        code = tl.where(x_abs >= 0.75, 2, code)
        # 1.25: even tie -> stays 1.0 (code 2)
        code = tl.where(x_abs > 1.25, 3, code)
        # 1.75: odd tie -> rounds up to 2.0 (code 4)
        code = tl.where(x_abs >= 1.75, 4, code)
        # 2.5: even tie -> stays 2.0 (code 4)
        code = tl.where(x_abs > 2.5, 5, code)
        # 3.5: odd tie -> rounds up to 4.0 (code 6)
        code = tl.where(x_abs >= 3.5, 6, code)
        # 5.0: even tie -> stays 4.0 (code 6)
        code = tl.where(x_abs > 5.0, 7, code)
        return code

    @triton.jit
    def _f32_to_e2m1_nibble(x):
        """Convert fp32 to a 4-bit e2m1 nibble (sign + 3 magnitude bits)."""
        sign = (x < 0).to(tl.int32)
        x_abs = tl.abs(x)
        mag = _f32_to_e2m1_unsigned_code(x_abs)
        return (sign << 3) | mag

    @triton.jit
    def _calc_scale_floor(amax):
        """FLOOR e8m0 scale, matching ScaleCalculationMode.FLOOR.

        Extract the floored power-of-two exponent of amax (in bf16 domain to
        match the torch reference), shift by the target max exponent for e2m1,
        clamp to e8m0 range, and return both the biased e8m0 byte and the
        reciprocal fp32 scale used to normalize the data.
        """
        # target max pow2 for e2m1 is the exponent of 6.0 ~ floor(log2(6)) = 2,
        # but torchao uses (max_pos exponent) which for e2m1 max (6.0) is 2.
        target_max_pow2: tl.constexpr = 2

        amax_bf16 = amax.to(tl.bfloat16)
        amax_i16 = amax_bf16.to(tl.int16, bitcast=True)
        # bf16: 1 sign, 8 exp, 7 mantissa
        bf16_mbits: tl.constexpr = 7
        bf16_exp_bias: tl.constexpr = 127
        extracted_pow2 = ((amax_i16 >> bf16_mbits) & 0xFF) - bf16_exp_bias
        extracted_pow2 = extracted_pow2 - target_max_pow2
        scale_unbiased = extracted_pow2.to(tl.int32)
        # clamp to representable e8m0 range; +1 to capture NaNs like the torch ref
        scale_unbiased = tl.maximum(
            tl.minimum(scale_unbiased, E8M0_EXP_BIAS + 1), -E8M0_EXP_BIAS
        )
        scale_e8m0_biased = (scale_unbiased + E8M0_EXP_BIAS).to(tl.uint8)

        # reciprocal scale in fp32: 2^(-scale_unbiased)
        descale = tl.exp2((-scale_unbiased).to(tl.float32))
        return scale_e8m0_biased, descale

    @triton.jit
    def _to_mxfp4_dim0_kernel(
        x_ptr,
        out_ptr,  # packed uint8, shape (n_rows, n_cols // 2)
        scale_ptr,  # uint8 e8m0, shape (n_rows, n_cols // BLOCK)
        n_rows,
        n_cols,
        ROW_TILE: tl.constexpr,
        COL_TILE: tl.constexpr,  # must be a multiple of BLOCK and even
        BLOCK: tl.constexpr,  # block size for scaling, 32 for MX
    ):
        pid_row = tl.program_id(0)
        pid_col = tl.program_id(1)

        BLOCKS_PER_COL_TILE: tl.constexpr = COL_TILE // BLOCK

        row_offs = pid_row * ROW_TILE + tl.arange(0, ROW_TILE)[:, None]
        col_offs = pid_col * COL_TILE + tl.arange(0, COL_TILE)[None, :]
        mask = (row_offs < n_rows) & (col_offs < n_cols)

        x = tl.load(x_ptr + row_offs.to(tl.int64) * n_cols + col_offs, mask=mask)
        x = x.to(tl.float32)

        # reshape into scaling blocks: (ROW_TILE * BLOCKS_PER_COL_TILE, BLOCK)
        x_blk = x.reshape(ROW_TILE * BLOCKS_PER_COL_TILE, BLOCK)
        amax = tl.max(tl.abs(x_blk), axis=1)
        scale_e8m0, descale = _calc_scale_floor(amax)
        x_norm = x_blk * descale[:, None]
        x_norm = x_norm.reshape(ROW_TILE, COL_TILE)

        # quantize to e2m1 nibbles
        nib = _f32_to_e2m1_nibble(x_norm)  # (ROW_TILE, COL_TILE), int32 in 0..15

        # pack two adjacent nibbles into one byte: even -> low, odd -> high.
        nib_flat = nib.reshape(ROW_TILE * COL_TILE)
        lo, hi = tl.split(nib_flat.reshape(ROW_TILE * COL_TILE // 2, 2))
        packed = ((hi << 4) | lo).to(tl.uint8)
        packed = packed.reshape(ROW_TILE, COL_TILE // 2)

        # store packed data
        out_row = pid_row * ROW_TILE + tl.arange(0, ROW_TILE)[:, None]
        out_col = pid_col * (COL_TILE // 2) + tl.arange(0, COL_TILE // 2)[None, :]
        out_mask = (out_row < n_rows) & (out_col < (n_cols // 2))
        tl.store(
            out_ptr + out_row.to(tl.int64) * (n_cols // 2) + out_col,
            packed,
            mask=out_mask,
        )

        # store e8m0 scales
        scale_cols = n_cols // BLOCK
        scale_e8m0_2d = scale_e8m0.reshape(ROW_TILE, BLOCKS_PER_COL_TILE)
        s_row = pid_row * ROW_TILE + tl.arange(0, ROW_TILE)[:, None]
        s_col = pid_col * BLOCKS_PER_COL_TILE + tl.arange(0, BLOCKS_PER_COL_TILE)[None, :]
        s_mask = (s_row < n_rows) & (s_col < scale_cols)
        tl.store(scale_ptr + s_row * scale_cols + s_col, scale_e8m0_2d, mask=s_mask)

    @triton_op("torchao::triton_to_mxfp4_dim0", mutates_args={})
    def triton_to_mxfp4_dim0(
        x: torch.Tensor,
        block_size: int = 32,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize ``x`` to mxfp4 across dim0 (rowwise blocks) with FLOOR scaling.

        Input:
          * ``x``: 2D contiguous tensor (bf16/fp16/fp32), n_cols % block_size == 0,
            n_cols even.
        Output:
          * ``data``:  ``float4_e2m1fn_x2`` packed data, shape (n_rows, n_cols // 2)
          * ``scale``: ``float8_e8m0fnu`` block scales, shape (n_rows, n_cols // block_size)
        """
        assert x.dim() == 2, "expected 2D input"
        assert x.is_contiguous(), "`x` must be contiguous"
        n_rows, n_cols = x.shape
        assert n_cols % block_size == 0, "n_cols must be divisible by block_size"
        assert n_cols % 2 == 0, "n_cols must be even for packing"

        out = torch.empty((n_rows, n_cols // 2), dtype=torch.uint8, device=x.device)
        scale = torch.empty(
            (n_rows, n_cols // block_size), dtype=torch.uint8, device=x.device
        )

        ROW_TILE = 1
        # process a whole row (or a chunk that is a multiple of block_size) per tile
        COL_TILE = n_cols

        grid = (
            triton.cdiv(n_rows, ROW_TILE),
            triton.cdiv(n_cols, COL_TILE),
        )
        wrap_triton(_to_mxfp4_dim0_kernel)[grid](
            x,
            out,
            scale,
            n_rows,
            n_cols,
            ROW_TILE=ROW_TILE,
            COL_TILE=COL_TILE,
            BLOCK=block_size,
        )
        return out.view(torch.float4_e2m1fn_x2), scale.view(torch.float8_e8m0fnu)

    @triton_to_mxfp4_dim0.register_fake
    def _(x: torch.Tensor, block_size: int = 32):
        n_rows, n_cols = x.shape
        data = x.new_empty((n_rows, n_cols // 2), dtype=torch.float4_e2m1fn_x2)
        scale = x.new_empty(
            (n_rows, n_cols // block_size), dtype=torch.float8_e8m0fnu
        )
        return data, scale

else:

    def triton_to_mxfp4_dim0(
        x: torch.Tensor, block_size: int = 32
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise AssertionError("needs triton")
