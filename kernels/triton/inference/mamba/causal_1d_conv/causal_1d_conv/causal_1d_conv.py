# Copyright (c) 2025, IBM Research

import torch
import triton
import triton.language as tl
from einops import rearrange
from typing import Literal, Optional

# vllm/attention/backends/utils.py
PAD_SLOT_ID = -1


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 256}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_stages=3, num_warps=8),
    ],
    key=["seqlen", "dim", "batch"],
)
@triton.jit()
def _causal_conv1d_fwd_kernel(
    # Pointers to matrices
    x_ptr,  # (batch, dim, seqlen)
    w_ptr,  # (dim, width)
    bias_ptr,
    initial_states_ptr,
    o_ptr,  # (batch, dim, seqlen)
    # Matrix dimensions
    batch,
    dim,
    seqlen,
    # Strides
    stride_x_seq,  # stride to get to next sequence,
    stride_x_dim,  # stride to get to next feature-value,
    stride_x_token,  # stride to get to next token (same feature-index, same sequence-index)
    stride_weight_dim,  # stride to get to next dim-axis value
    stride_weight_width,  # stride to get to next width-axis value
    stride_istate_seq,
    stride_istate_dim,
    stride_istate_token,
    stride_o_seq,
    stride_o_dim,
    stride_o_token,
    # Meta-parameters
    HAS_BIAS: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,  # maybe using this we don't need 'width'
    SILU_ACTIVATION: tl.constexpr,
    HAS_INITIAL_STATES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    indices_0 = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    idx_seqs = indices_0 // seqlen
    idx_tokens = indices_0 % seqlen

    x_base = x_ptr + (idx_seqs * stride_x_seq)[:, None]  # the beginning features at all tokens at all sequences processed by this Triton program
    idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    w_base = w_ptr + (idx_feats * stride_weight_dim)  # first kernel column, configured for weights to handle BLOCK_N features in range
    load_init_state = False
    if HAS_INITIAL_STATES:
        load_init_state = tl.min(idx_tokens) < KERNEL_WIDTH - 1
        initial_states_base = initial_states_ptr + (idx_seqs * stride_istate_seq)[:, None] + (idx_feats * stride_istate_dim)[None, :]

    # store output data at the corresponding tokens (BLOCK_M of them) and feature-indices (BLOCK_N of them) in these tokens
    if HAS_BIAS:
        bias = bias_ptr + idx_feats
        mask_bias = idx_feats < dim
        acc = tl.load(bias, mask=mask_bias, other=0.0).to(tl.float32)[None, :]  # [BLOCK_N]
        acc = tl.broadcast_to(acc, (BLOCK_M, BLOCK_N))
    else:
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    PADDING_W = KERNEL_WIDTH - 1
    for j in range(KERNEL_WIDTH):
        idx_x_w = j - PADDING_W + idx_tokens  # the token index to multiply with kernel[:, 0], given kernel with width-columns, i.e. kernel[:, 0..(width-1)]
        x_ptrs = x_base + ((idx_x_w * stride_x_token)[:, None] + (idx_feats * stride_x_dim)[None, :])  # [BLOCK_M, BLOCK_N]
        mask_x = ((idx_seqs < batch)[:, None]  # sequence-index
                  & (idx_x_w >= 0)[:, None]  # token-index
                  & (idx_x_w < seqlen)[:, None]  # token-index
                  & (idx_feats < dim)[None, :]  # feature-index
                  )
        if HAS_INITIAL_STATES:
            if load_init_state:
                initial_states_ptrs = initial_states_base + ((idx_x_w + KERNEL_WIDTH - 1) * stride_istate_token)[:, None]  # [BLOCK_M, BLOCK_N]
                mask_w = (idx_seqs < batch)[:, None] & (idx_x_w < 0)[:, None] & (idx_feats < dim)[None, :]  # sequence-index  # token-index  # feature-index
                initial_states = tl.load(initial_states_ptrs, mask_w, 0.0)
            else:
                initial_states = tl.zeros((BLOCK_M, BLOCK_N), dtype=x_ptr.dtype.element_ty)
            matrix_x = tl.load(x_ptrs, mask=mask_x, other=initial_states)
        else:
            matrix_x = tl.load(x_ptrs, mask=mask_x, other=0.0)

        w_ptrs = w_base[None, :] + \
            (j * stride_weight_width)  # [1, BLOCK_N] tensor
        mask_w = (idx_feats < dim)[None, :]
        matrix_w = tl.load(w_ptrs, mask_w, other=0.0)
        acc += matrix_x * matrix_w

    if SILU_ACTIVATION:
        acc = acc / (1 + tl.exp(-acc))
    mask = (
        (idx_seqs < batch)[:, None]  # sequence-index
        & (idx_tokens < seqlen)[:, None]  # token-index
        & (idx_feats < dim)[None, :]  # feature-index
    )
    o_ptrs = (
        o_ptr
        + (idx_seqs * stride_o_seq)[:, None]
        + (idx_tokens * stride_o_token)[:, None]
        + (idx_feats * stride_o_dim)[None, :]
    )

    tl.store(o_ptrs, acc, mask=mask)


def causal_conv1d_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    seq_idx: Optional[torch.Tensor] = None,
    initial_states: Optional[torch.Tensor] = None,
    return_final_states: Optional[torch.Tensor] = False,
    final_states_out: Optional[torch.Tensor] = None,
    activation: Optional[Literal["silu", "swish"]] = None,
):
    batch, dim, seqlen = x.shape
    _, width = weight.shape
    assert (dim, width) == weight.shape
    assert x.stride(2) == 1 or x.stride(1) == 1
    # TODO: we may want to use weight such that weight.stride(dim)==1
    assert weight.stride(1) == 1
    # Tensor layout as NHWC is called channel last with 'C' is time-dimension
    is_channel_last = (x.stride(1) == 1) & (x.stride(2) > 1)
    stride_w_dim = weight.stride(0)
    stride_w_width = weight.stride(1)
    # effort to make data contiguous along dim-axis:
    weight = weight.transpose(0, 1).contiguous()
    stride_w_dim = weight.stride(1)
    stride_w_width = weight.stride(0)

    # assert initial_states is None  # only this for now
    assert return_final_states is False
    stride_istate_seq = 0
    stride_istate_dim = 0
    stride_istate_token = 0
    if initial_states is not None:
        assert (batch, dim, width - 1) == initial_states.shape
        stride_istate_seq = initial_states.stride(0)
        stride_istate_dim = initial_states.stride(1)
        stride_istate_token = initial_states.stride(2)
        assert stride_istate_dim == 1

    out = torch.empty_like(x)

    if not is_channel_last:
        assert 0, "Need to run in channel-last layout"
    else:

        def grid(META):
            return (
                triton.cdiv(batch * seqlen, META["BLOCK_M"]),
                triton.cdiv(dim, META["BLOCK_N"]),
            )

        with torch.cuda.device(x.device.index):
            _causal_conv1d_fwd_kernel[grid](
                # Pointers to matrices
                x,
                weight,
                bias,
                initial_states,
                out,
                # Matrix dimensions
                batch,
                dim,
                seqlen,
                # stride
                x.stride(0),
                x.stride(1),
                x.stride(2),
                stride_w_dim,
                stride_w_width,
                stride_istate_seq,
                stride_istate_dim,
                stride_istate_token,
                out.stride(0),
                out.stride(1),
                out.stride(2),
                # META
                HAS_BIAS=bias is not None,
                KERNEL_WIDTH=width,
                SILU_ACTIVATION=activation in ["silu", "swish"],
                HAS_INITIAL_STATES=initial_states is not None,
            )
    return out


class CausalConv1dFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        weight,
        bias=None,
        seq_idx=None,
        initial_states=None,
        return_final_states: bool = False,
        final_states_out=None,
        activation: Optional[Literal["silu", "swish"]] = None,
    ):
        # NOTE: in fact, 'beta=1' would turn swish into silu - and only silu form is used
        if x.stride(2) != 1 and x.stride(1) != 1:
            x = x.contiguous()
        bias = bias.contiguous() if bias is not None else None
        if seq_idx is not None:
            assert initial_states is None, "initial_states must be None if seq_idx is not None"
            assert not return_final_states, "If seq_idx is not None, we don't return final_states_out"
        seq_idx = seq_idx.contiguous() if seq_idx is not None else None
        if initial_states is not None and ((initial_states.stride(2) != 1) and (initial_states.stride(1) != 1)):
            initial_states = initial_states.contiguous()
        if return_final_states:
            assert (
                x.stride(1) == 1
            ), "Only channel-last layout support returning final_states_out"
            if final_states_out is not None:
                assert (
                    (final_states_out.stride(2) == 1) or (
                        final_states_out.stride(1) == 1)
                )
            else:
                batch, dim, seqlen = x.shape
                width = weight.shape[1]
                final_states_out = torch.empty(
                    batch, width - 1, dim, device=x.device, dtype=x.dtype).transpose(1, 2)
        else:
            final_states_out = None
        ctx.activation = activation
        out = causal_conv1d_fwd(
            x,
            weight,
            bias=bias,
            seq_idx=seq_idx,
            initial_states=initial_states,
            return_final_states=return_final_states,
            final_states_out=final_states_out,
            activation=ctx.activation,
        )
        ctx.save_for_backward(x, weight, bias, seq_idx, initial_states)
        ctx.return_final_states = return_final_states
        ctx.return_dinitial_states = initial_states is not None and initial_states.requires_grad
        return out if not return_final_states else (out, final_states_out)

    @staticmethod
    def backward(ctx, dout, *args):
        """dout = dL/dy
        RETURN: dL/dx, dL/dweight, dL/dbias, ...
        GIVEN THAT: def forward(ctx, x, weight, bias=None...)
        """
        x, weight, bias, seq_idx, initial_states = ctx.saved_tensors
        dfinal_states = args[0] if ctx.return_final_states else None
        if dout.stride(2) != 1 and dout.stride(1) != 1:
            dout = dout.contiguous()
        # The kernel supports passing in a pre-allocated dx (e.g., in case we want to fuse the
        # backward of conv1d with the backward of chunk).
        # Here we just pass in None and dx will be allocated in the C++ code.
        dx, dweight, dbias, dinitial_states = causal_conv1d_bwd(
            x,
            weight,
            bias,
            dout,
            seq_idx,
            initial_states,
            dfinal_states,
            None,
            ctx.return_dinitial_states,
            ctx.activation,
        )
        return (
            dx,
            dweight,
            dbias if bias is not None else None,
            None,
            dinitial_states if initial_states is not None else None,
            None,
            None,
            None,
        )


def causal_conv1d_fn(
    x,  # channel last, i.e. (batch, dim, seqlen)
    weight,  # (dim, w)
    bias=None,  # (dim, )scalar
    seq_idx=None,
    initial_states=None,
    return_final_states=False,
    final_states_out=None,
    activation: Optional[Literal["silu", "swish"]] = None,
):
    """causal_conv1d_fn.

    :param x: (batch, dim, seqlen) tensor
    :param weight: (dim, w) tensor
    :param bias: (dim,) tensor
    :param activation: ["silu", "swish"]
    :param seq_idx=None
    :param initial_states=None
    :param return_final_states=False
    :param final_states_out=None

    Return: (batch, dim, seqlen) tensor
    """
    if weight.dim() == 3:
        assert weight.shape[1] == 1
        weight = rearrange(weight, "d 1 w -> d w")
    return CausalConv1dFn.apply(
        x,
        weight,
        bias,
        seq_idx,
        initial_states,
        return_final_states,
        final_states_out,
        activation,
    )


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 128}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_N": 64}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_N": 32}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_N": 16}, num_stages=3, num_warps=8),
    ],
    key=["dim"],
    restore_value=["conv_state_ptr", "x_ptr"],
)
@triton.jit()
def _causal_conv1d_update_kernel(
    # Pointers to matrices
    x_ptr,  # (batch, dim, seqlen)
    w_ptr,  # (dim, width)
    bias_ptr,
    conv_state_ptr,
    cache_seqlens_ptr,
    conv_state_indices_ptr,
    o_ptr,  # (batch, dim, seqlen)
    # Matrix dimensions
    batch,
    dim,
    seqlen,
    state_len,
    num_cache_lines,
    # Strides
    stride_x_seq,  # stride to get to next sequence,
    stride_x_dim,  # stride to get to next feature-value,
    stride_x_token,  # stride to get to next token (same feature-index, same sequence-index)
    stride_weight_dim,  # stride to get to next dim-axis value
    stride_weight_width,  # stride to get to next width-axis value
    stride_conv_state_seq,
    stride_conv_state_dim,
    stride_conv_state_tok,
    stride_o_seq,
    stride_o_dim,
    stride_o_token,
    # others
    pad_slot_id,
    # Meta-parameters
    HAS_BIAS: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    IS_CIRCULAR_BUFFER: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    idx_seq = tl.program_id(0)
    if idx_seq >= batch:
        return

    idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)

    w_base = w_ptr + (idx_feats * stride_weight_dim)
    if IS_CIRCULAR_BUFFER:
        cache_seqlen = tl.load(cache_seqlens_ptr + idx_seq)  # modulo later
    else:
        cache_seqlen = 0
    # store output data at the corresponding tokens (BLOCK_M of them) and feature-indices (BLOCK_N of them) in these tokens
    if IS_CONTINUOUS_BATCHING:
        conv_state_batch_coord = tl.load(conv_state_indices_ptr + idx_seq)
    else:
        conv_state_batch_coord = idx_seq
    if USE_PAD_SLOT:
        if conv_state_batch_coord == pad_slot_id:
            # not processing
            return
    conv_state_base = (
        conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (idx_feats * stride_conv_state_dim)
    )  # [BLOCK_N,]

    for idx_token in range(seqlen):
        x_base = x_ptr + (idx_seq * stride_x_seq) + (idx_feats * stride_x_dim)  # [BLOCK_N, ]

        if HAS_BIAS:
            bias = bias_ptr + idx_feats
            mask_bias = idx_feats < dim
            acc = tl.load(bias, mask=mask_bias, other=0.0).to(tl.float32)  # [BLOCK_N]
        else:
            acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        PADDING_W = KERNEL_WIDTH - 1
        for j in range(KERNEL_WIDTH):
            # the token index to multiply with kernel[:, 0], given kernel with width-columns, i.e. kernel[:, 0..(width-1)]
            idx_x_w = j - PADDING_W + idx_token
            x_ptrs = x_base + (idx_x_w * stride_x_token)  # [BLOCK_N]
            mask_x = (idx_x_w >= 0) & (idx_x_w < seqlen) & (idx_feats < dim)
            if IS_CIRCULAR_BUFFER:
                assert 0  # TUAN TODO: double check the logic - it seems correct
                conv_state_ptrs = (
                    conv_state_base + (((idx_x_w + cache_seqlen) % state_len) * stride_conv_state_tok)[:, None]
                )  # [BLOCK_M, BLOCK_N]
            else:
                conv_state_ptrs = conv_state_base + ((idx_x_w + state_len) * stride_conv_state_tok)  # [BLOCK_N]
                mask_w = (conv_state_batch_coord < num_cache_lines) & (idx_x_w < 0) & (idx_feats < dim)
                conv_state = tl.load(conv_state_ptrs, mask_w, 0.0)
            matrix_x = tl.load(x_ptrs, mask=mask_x, other=conv_state)

            w_ptrs = w_base + (j * stride_weight_width)  # [BLOCK_N] tensor
            mask_w = idx_feats < dim
            matrix_w = tl.load(w_ptrs, mask_w, other=0.0)
            acc += matrix_x * matrix_w  # [BLOCK_N]

        if SILU_ACTIVATION:
            acc = acc / (1 + tl.exp(-acc))
        mask = (idx_token < seqlen) & (idx_feats < dim)  # sequence-index  # token-index  # feature-index
        o_ptrs = o_ptr + (idx_seq * stride_o_seq) + (idx_token * stride_o_token) + (idx_feats * stride_o_dim)
        tl.store(o_ptrs, acc, mask=mask)

    if IS_CIRCULAR_BUFFER:
        # TODO:
        assert 0
    else:
        idx_tokens = tl.arange(0, NP2_STATELEN)  # [BLOCK_M]

        conv_state_ptrs_source = (
            conv_state_ptr
            + (conv_state_batch_coord * stride_conv_state_seq)
            + (idx_feats * stride_conv_state_dim)[None, :]
            + ((idx_tokens + seqlen) * stride_conv_state_tok)[:, None]
        )  # [BLOCK_M, BLOCK_N]
        mask = (
            (conv_state_batch_coord < num_cache_lines)
            & ((idx_tokens + seqlen) < state_len)[:, None]
            & (idx_feats < dim)[None, :]
        )
        conv_state = tl.load(conv_state_ptrs_source, mask, other=0.0)

        VAL = state_len - seqlen
        x_base = x_ptr + (idx_seq * stride_x_seq) + (idx_feats * stride_x_dim)[None, :]  # [1, BLOCK_N]

        x_ptrs = x_base + ((idx_tokens - VAL) * stride_x_token)[:, None]  # [BLOCK_M, BLOCK_N]

        mask_x = (
            (idx_tokens - VAL >= 0)[:, None] & (idx_tokens - VAL < seqlen)[:, None] & (idx_feats < dim)[None, :]
        )  # token-index  # token-index  # feature-index
        loaded_x = tl.load(x_ptrs, mask_x, 0.0)
        tl.debug_barrier()

        new_conv_state = tl.where(mask, conv_state, loaded_x)
        conv_state_ptrs_target = conv_state_base + (idx_tokens * stride_conv_state_tok)[:, None]  # [BLOCK_M, BLOCK_N]
        mask = (idx_tokens < state_len)[:, None] & (idx_feats < dim)[None, :]
        tl.store(conv_state_ptrs_target, new_conv_state, mask)


def causal_conv1d_update(
    x,
    conv_state,
    weight,
    bias=None,
    activation: Optional[Literal["silu", "swish"]] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    conv_state_indices: Optional[torch.Tensor] = None,
    pad_slot_id: int = None,
):
    """
    x: (batch, dim) or (batch, dim, seqlen)
        new tokens whose causal-conv-1d need to be computed
    conv_state: (..., dim, state_len), where state_len >= width - 1
        (function as `init_state` in causal_conv1d_fn API)
        hold the previous `state_len` tokens that we can use to compute causal-conv-1d of new tokens from 'x'
            * if `conv_sate_indices` is provided: behave like continuous batching mode
            * if `cache_seqlens` is provided: also behave like a circular buffer
        ==============
        [in standard batching, naturally we expect conv_state[i] is used for x[i] with i is sequence-index
        [in continuous batching, the corresponding prior data for sequence x[i] is
           NOT NECESSARY from conv_state[i];
           BUT CAN BE from conv_state[conv_state_indices[i]]
         given i=batch_id=sequence_id
        IN OTHER WORDS: conv_state[j] | x[i]
          with j = i [if conv_state_indices is NOne
          with j = conv_state_indices[i] otherwise
        ]
        [NOTE: can be used as a circular buffer if `cache_seqlens` is provided]
    weight: (dim, width)
        (causal) 1d conv kernel
    bias: (dim,)
    cache_seqlens: (batch,), dtype int32.
        [ PRIOR:
          i.e. [conv_state[j][k] | x[i][0] ]
        ]
        Hold the token-index (3rd axis) in the `conv_state` where we ...
        If not None, the conv_state is treated as a circular buffer.
        The conv_state will be updated by copying x to the conv_state starting at the index
        @cache_seqlens % state_len.
    conv_state_indices: (batch,), dtype int32
        If present, then it is used to extract the row in `conv_state` to be used with corresponding sequence x[i]
        i.e. for the given sequence i-th, and j-th is the index in `conv_state` where to get data to combine with x[i] for computing causal-1d-conv
          j = i if conv_state_indices is None
          j = conv_state_indices[i] otherwise
        i.e. [conv_state[j] | x[i] ]
        Useful for a continuous batching scenario.
    pad_slot_id: int | None
        If used, the constant value that we can use to compare with conv_state_indices[i], if
        conv_state_indices[i] == pad_slot_id, then we ignore data from that row of conv_state[conv_state_indices[i]]

    out: (batch, dim) or (batch, dim, seqlen)
    """
    unsqueeze = x.dim() == 2
    if unsqueeze:
        # make it (batch, dim, seqlen) with seqlen == 1
        x = x.unsqueeze(-1)
    batch, dim, seqlen = x.shape
    _, width = weight.shape

    # conv_state: (..., dim, state_len), where state_len >= width - 1
    state_len = conv_state.size(2)
    assert state_len >= width - 1
    assert dim == conv_state.size(1)
    if conv_state_indices is None:
        assert conv_state.size(0) >= batch
    else:
        assert (batch,) == conv_state_indices.shape
    num_cache_lines = conv_state.size(0)

    stride_w_dim = weight.stride(0)
    stride_w_width = weight.stride(1)

    def grid(META):
        return (
            batch,
            triton.cdiv(dim, META["BLOCK_N"]),
        )

    assert cache_seqlens is None  # TUAN: FOR NOW (not needed for vLLM) - circular buffer # fmt:off
    out = torch.empty_like(x)
    with torch.cuda.device(x.device.index):
        _causal_conv1d_update_kernel[grid](
            # Pointers to matrices
            x,
            weight,
            bias,
            conv_state,
            cache_seqlens,
            conv_state_indices,
            out,
            # Matrix dimensions
            batch,
            dim,
            seqlen,
            state_len,
            num_cache_lines,
            # stride
            x.stride(0),  # X (batch, dim, seqlen)
            x.stride(1),
            x.stride(2),
            stride_w_dim,
            stride_w_width,
            conv_state.stride(0),
            conv_state.stride(1),
            conv_state.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            # others
            pad_slot_id,
            # META
            HAS_BIAS=bias is not None,
            KERNEL_WIDTH=width,
            SILU_ACTIVATION=activation in ["silu", "swish"],
            IS_CONTINUOUS_BATCHING=conv_state_indices is not None,
            IS_CIRCULAR_BUFFER=cache_seqlens is not None,
            NP2_STATELEN=triton.next_power_of_2(state_len),
            USE_PAD_SLOT=pad_slot_id is not None,
        )
    if unsqueeze:
        out = out.squeeze(-1)
    return out


def causal_conv1d_update_vllm(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Optional[Literal["silu", "swish"]] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    conv_state_indices: Optional[torch.Tensor] = None,
    pad_slot_id: int = PAD_SLOT_ID,
):
    """
    x: (batch, dim) or (batch, dim, seqlen)
        [shape=2: single token prediction]
        [shape=3: multiple tokens prediction]
    conv_state: (batch, dim, state_len), where state_len >= width - 1
    weight: (dim, width)
    bias: (dim,)
    cache_seqlens: (batch,), dtype int32.
        If not None, the conv_state is treated as a circular buffer.
        The conv_state will be updated by copying x to the conv_state
        starting at the index
        @cache_seqlens % state_len.
    conv_state_indices: (batch,), dtype int32
        If not None, the conv_state is a larger tensor along the batch dim,
        and we are selecting the batch coords specified by conv_state_indices.
        Useful for a continuous batching scenario.
    pad_slot_id: int
            if cache_indices is passed, lets the kernel identify padded
            entries that will not be processed,
            for example: cache_indices = [pad_slot_id, 1 ,20 ,pad_slot_id]
            in this case, the kernel will not process entries at
            indices 0 and 3
    out: (batch, dim) or (batch, dim, seqlen)
    """
    assert cache_seqlens is None
    # TODO : adopt the strategy in vLLM that overwrite on 'x' directly, rather than creating a new tensor 'o'
    o = causal_conv1d_update(
        x,
        conv_state,
        weight,
        bias=bias,
        activation=activation,
        cache_seqlens=cache_seqlens,
        conv_state_indices=conv_state_indices,
        pad_slot_id=pad_slot_id,
    )
    return o
