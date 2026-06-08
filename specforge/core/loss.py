"""
This file incorporates code from Unsloth licensed under the Apache License, Version 2.0.
See the original Unsloth repository at https://github.com/unslothai/unsloth.
The idea of in-place backward pass is from Liger-Kernel.
See the original Liger-Kernel repository at https://github.com/linkedin/Liger-Kernel.

Two implementations coexist:
- LogSoftmaxLoss: GPU (CUDA) implementation, original Unsloth/Liger 2-pass kernels
- LogSoftmaxLossNPU: Ascend NPU implementation, single-pass optimized kernels
  Memory optimization: saves only m and d (B*T scalars) instead of full probs (B*T*V).
  Performance optimizations:
  1. Forward single-pass — accumulate sum_target_logits + sum_target alongside m/d,
     compute loss = -(sum_target_logits - sum_target * (m + log(d)))
  2. Backward single-pass — precompute target_grad_sum on host, eliminating pass 1
  3. grad_output_scaled passed as float param, eliminating per-row scalar load
  4. care_padding=False on all loads (5-10% improvement per Ascend guidelines)
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl

import torch.nn.functional as F

_NPU_AVAILABLE = False
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    _NPU_AVAILABLE = True
except ImportError:
    pass

class PyTorchNativeLogSoftmaxLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, target, position_mask):
        log_probs = F.log_softmax(logits, dim=-1)
        plogp = target * log_probs
        loss = -torch.sum(position_mask * plogp, dim=2).mean()
        probs = torch.exp(log_probs)
        ctx.save_for_backward(probs, target, position_mask)

        return loss

    @staticmethod
    def backward(ctx, grad_output):
        probs, target, position_mask = ctx.saved_tensors
        B, T, V = probs.shape

        scaling_factor = grad_output / (B * T)

        target_sum = target.sum(dim=-1, keepdim=True)

        grad_logits = (probs * target_sum - target)
        grad_logits = grad_logits * position_mask
        grad_logits = grad_logits * scaling_factor
        return grad_logits, None, None

def pytorch_native_compute_loss(logits, target, position_mask):
    return PyTorchNativeLogSoftmaxLoss.apply(logits, target, position_mask)

# Reference implementation — torch.compile only on CUDA (Inductor generates UB-unsafe Triton on NPU)
def _compute_loss(logits, target_p, position_mask):
    logits = logits.float()
    out_logp = nn.LogSoftmax(dim=2)(logits)
    plogp = target_p * out_logp
    loss = -torch.sum(position_mask * plogp, 2).mean()
    return loss

if not _NPU_AVAILABLE:
    _compute_loss = torch.compile(dynamic=None)(_compute_loss)


def _calculate_settings(n):
    # reference: https://github.com/unslothai/unsloth/blob/fd753fed99ed5f10ef8a9b7139588d9de9ddecfb/unsloth/kernels/utils.py#L43

    MAX_FUSED_SIZE = 131072
    BLOCK_SIZE = triton.next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(
            f"Cannot launch Triton kernel since n = {n} exceeds the recommended Triton blocksize = {MAX_FUSED_SIZE}."
        )

    num_warps = 4
    if BLOCK_SIZE >= 32768:
        num_warps = 32
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8

    # AMD GPU (ROCm)
    if hasattr(torch.version, "hip") and torch.version.hip is not None:
        num_warps //= 2

    return BLOCK_SIZE, num_warps


@triton.jit
def log_softmax_forward_kernel(
    logits_ptr,
    logits_stride,
    target_ptr,
    target_stride,
    position_mask_ptr,
    position_mask_stride,
    loss_ptr,
    loss_stride,
    m_ptr,
    d_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    program_id = tl.program_id(0).to(tl.int64)
    logits_ptr += program_id * logits_stride
    target_ptr += program_id * target_stride
    position_mask_ptr += program_id * position_mask_stride
    position_mask = tl.load(position_mask_ptr)
    if position_mask == 0:
        return

    m = float("-inf")
    d = 0.0

    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        logits_block = tl.load(
            logits_ptr + offsets, mask=mask, other=float("-inf")
        ).cast(tl.float32)
        block_max = tl.max(tl.where(mask, logits_block, float("-inf")))
        m_new = tl.maximum(m, block_max)
        d = d * tl.exp(m - m_new) + tl.sum(
            tl.where(mask, tl.exp(logits_block - m_new), 0.0)
        )
        m = m_new

    loss = 0.0
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        logits_block = tl.load(logits_ptr + offsets, mask=mask, other=0.0).cast(
            tl.float32
        )
        target_block = tl.load(target_ptr + offsets, mask=mask, other=0.0).cast(
            tl.float32
        )
        # log-softmax: log(exp(x - max) / sum) = (x - max) - log(sum)
        normalized_logits = logits_block - m
        log_normalizer = tl.log(d)
        log_softmax_logits = normalized_logits - log_normalizer
        weighted_log_prob = target_block * log_softmax_logits
        loss += tl.sum(tl.where(mask, weighted_log_prob, 0.0))

    loss_ptr += program_id * loss_stride
    m_ptr += program_id
    d_ptr += program_id
    tl.store(loss_ptr, -loss)
    tl.store(m_ptr, m.to(tl.float32))
    tl.store(d_ptr, d.to(tl.float32))


@triton.jit
def log_softmax_backward_kernel(
    logits_ptr,
    logits_stride,
    target_ptr,
    target_stride,
    position_mask_ptr,
    grad_output_ptr,
    scaling_factor,
    m_ptr,
    d_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    program_id = tl.program_id(0).to(tl.int64)
    logits_ptr += program_id * logits_stride
    target_ptr += program_id * target_stride
    position_mask_ptr += program_id

    position_mask = tl.load(position_mask_ptr)
    if position_mask == 0:
        for i in range(0, n_cols, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_cols
            tl.store(logits_ptr + offsets, 0.0, mask=mask)
        return

    m_ptr += program_id
    d_ptr += program_id
    m = tl.load(m_ptr).to(tl.float32)
    d = tl.load(d_ptr).to(tl.float32)
    grad_output = tl.load(grad_output_ptr).to(tl.float32)
    grad_output = grad_output * scaling_factor

    # First pass: compute sum of (target * grad_output)
    target_grad_sum = 0.0
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        target_block = tl.load(target_ptr + offsets, mask=mask, other=0.0).cast(
            tl.float32
        )
        target_grad_sum += tl.sum(tl.where(mask, target_block * grad_output, 0.0))

    # Second pass: compute log-softmax gradients
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        logits_block = tl.load(logits_ptr + offsets, mask=mask, other=0.0).cast(
            tl.float32
        )
        target_block = tl.load(target_ptr + offsets, mask=mask, other=0.0).cast(
            tl.float32
        )
        softmax_prob = tl.exp(logits_block - m) / d
        normalized_grad = softmax_prob * target_grad_sum
        grad_block = -(target_block * grad_output - normalized_grad)
        tl.store(logits_ptr + offsets, grad_block.to(tl.float32), mask=mask)


class LogSoftmaxLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, target, position_mask):
        B, T, V = logits.shape
        loss = torch.zeros((B * T, 1), device=logits.device)
        logits_flat = logits.contiguous().view(B * T, V)
        target_flat = target.contiguous().view(B * T, V)
        position_mask_flat = position_mask.contiguous().view(B * T, 1).bool()
        grid = (B * T,)
        m = torch.zeros((B * T,), device=logits.device, dtype=torch.float32)
        d = torch.zeros((B * T,), device=logits.device, dtype=torch.float32)
        BLOCK_SIZE, num_warps = _calculate_settings(V)
        log_softmax_forward_kernel[grid](
            logits_flat,
            logits_flat.stride(0),
            target_flat,
            target_flat.stride(0),
            position_mask_flat,
            position_mask_flat.stride(0),
            loss,
            loss.stride(0),
            m,
            d,
            V,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        ctx.save_for_backward(logits.detach(), target, position_mask, m, d)
        return loss.squeeze(1).mean()

    @staticmethod
    def backward(ctx, grad_output):
        logits, target, position_mask, m, d = ctx.saved_tensors
        B, T, V = logits.shape
        scaling_factor = 1.0 / (B * T)
        logits = logits.contiguous().view(B * T, V)
        target = target.contiguous().view(B * T, V)
        position_mask = position_mask.contiguous().view(B * T, 1).bool()
        grid = (B * T,)
        BLOCK_SIZE, num_warps = _calculate_settings(V)
        log_softmax_backward_kernel[grid](
            logits,
            logits.stride(0),
            target,
            target.stride(0),
            position_mask,
            grad_output,
            scaling_factor,
            m,
            d,
            V,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        logits = logits.view(B, T, V)
        return logits, None, None, None, None


# ============================================================
# Ascend NPU branch — single-pass optimized kernels
# ============================================================

def _calculate_settings_npu(n):
    NPU_MAX_BLOCK_SIZE = 4096
    BLOCK_SIZE = min(triton.next_power_of_2(n), NPU_MAX_BLOCK_SIZE)
    return BLOCK_SIZE


@triton.jit
def log_softmax_forward_kernel_npu(
    logits_ptr,
    logits_stride,
    target_ptr,
    target_stride,
    position_mask_ptr,
    position_mask_stride,
    loss_ptr,
    loss_stride,
    m_ptr,
    d_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    program_id = tl.program_id(0).to(tl.int64)
    logits_ptr += program_id * logits_stride
    target_ptr += program_id * target_stride
    position_mask_ptr += program_id * position_mask_stride
    position_mask = tl.load(position_mask_ptr)
    if position_mask == 0:
        return

    m = float("-inf")
    d = 0.0
    sum_target_logits = 0.0
    sum_target = 0.0

    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        logits_block = tl.load(
            logits_ptr + offsets, mask=mask, other=float("-inf"), care_padding=False
        ).cast(tl.float32)
        target_block = tl.load(
            target_ptr + offsets, mask=mask, other=0.0, care_padding=False
        ).cast(tl.float32)

        block_max = tl.max(tl.where(mask, logits_block, float("-inf")))
        m_new = tl.maximum(m, block_max)
        d = d * tl.exp(m - m_new) + tl.sum(
            tl.where(mask, tl.exp(logits_block - m_new), 0.0)
        )
        m = m_new

        sum_target_logits += tl.sum(tl.where(mask, target_block * logits_block, 0.0))
        sum_target += tl.sum(tl.where(mask, target_block, 0.0))

    loss = -(sum_target_logits - sum_target * (m + tl.log(d)))

    loss_ptr += program_id * loss_stride
    m_ptr += program_id
    d_ptr += program_id
    tl.store(loss_ptr, loss)
    tl.store(m_ptr, m.to(tl.float32))
    tl.store(d_ptr, d.to(tl.float32))


@triton.jit
def log_softmax_backward_kernel_npu(
    logits_ptr,
    logits_stride,
    target_ptr,
    target_stride,
    position_mask_ptr,
    target_grad_sum_ptr,
    m_ptr,
    d_ptr,
    grad_output_scaled,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    program_id = tl.program_id(0).to(tl.int64)
    logits_ptr += program_id * logits_stride
    target_ptr += program_id * target_stride
    position_mask_ptr += program_id

    position_mask = tl.load(position_mask_ptr)
    if position_mask == 0:
        for i in range(0, n_cols, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_cols
            tl.store(logits_ptr + offsets, 0.0, mask=mask)
        return

    m_ptr += program_id
    d_ptr += program_id
    target_grad_sum_ptr += program_id
    m = tl.load(m_ptr).to(tl.float32)
    d = tl.load(d_ptr).to(tl.float32)
    target_grad_sum = tl.load(target_grad_sum_ptr).to(tl.float32)

    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        logits_block = tl.load(
            logits_ptr + offsets, mask=mask, other=0.0, care_padding=False
        ).cast(tl.float32)
        target_block = tl.load(
            target_ptr + offsets, mask=mask, other=0.0, care_padding=False
        ).cast(tl.float32)
        softmax_prob = tl.exp(logits_block - m) / d
        normalized_grad = softmax_prob * target_grad_sum
        grad_block = -(target_block * grad_output_scaled - normalized_grad)
        tl.store(logits_ptr + offsets, grad_block.to(tl.float32), mask=mask)


class LogSoftmaxLossNPU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, target, position_mask):
        B, T, V = logits.shape
        loss = torch.zeros((B * T, 1), device=logits.device)
        logits_flat = logits.contiguous().view(B * T, V)
        target_flat = target.contiguous().view(B * T, V)
        position_mask_flat = position_mask.contiguous().view(B * T, 1).bool()
        grid = (B * T,)
        m = torch.zeros((B * T,), device=logits.device, dtype=torch.float32)
        d = torch.zeros((B * T,), device=logits.device, dtype=torch.float32)
        BLOCK_SIZE = _calculate_settings_npu(V)
        log_softmax_forward_kernel_npu[grid](
            logits_flat,
            logits_flat.stride(0),
            target_flat,
            target_flat.stride(0),
            position_mask_flat,
            position_mask_flat.stride(0),
            loss,
            loss.stride(0),
            m,
            d,
            V,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        ctx.save_for_backward(logits.detach(), target, position_mask, m, d)
        return loss.squeeze(1).mean()

    @staticmethod
    def backward(ctx, grad_output):
        logits, target, position_mask, m, d = ctx.saved_tensors
        B, T, V = logits.shape
        logits = logits.contiguous().view(B * T, V)
        target = target.contiguous().view(B * T, V)
        position_mask = position_mask.contiguous().view(B * T, 1).bool()

        scaling_factor = 1.0 / (B * T)
        grad_output_scaled = grad_output.item() * scaling_factor
        target_sum_per_row = target.sum(dim=-1)
        target_grad_sum_per_row = target_sum_per_row * grad_output_scaled

        grid = (B * T,)
        BLOCK_SIZE = _calculate_settings_npu(V)
        log_softmax_backward_kernel_npu[grid](
            logits,
            logits.stride(0),
            target,
            target.stride(0),
            position_mask,
            target_grad_sum_per_row,
            m,
            d,
            grad_output_scaled,
            V,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        logits = logits.view(B, T, V)
        return logits, None, None


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "npu"
    B, T, V = 1, 1024, 16000
    logits = torch.randn(B, T, V, device=device, requires_grad=True)
    logits2 = logits.clone().detach().requires_grad_(True)
    target = torch.randn(B, T, V, device=device)
    position_mask = torch.ones((B, T, 1), dtype=torch.bool, device=device)

    if device == "npu":
        loss_fn = LogSoftmaxLossNPU.apply
    else:
        loss_fn = LogSoftmaxLoss.apply

    output1 = LogSoftmaxLossNPU.apply(logits, target, position_mask)
    output2 = _compute_loss(logits2, target, position_mask)
    torch.testing.assert_close(output1, output2, rtol=1e-4, atol=1e-4)
    output1.backward()
    output2.backward()
    torch.testing.assert_close(logits.grad, logits2.grad, rtol=1e-4, atol=1e-4)
