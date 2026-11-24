"""
v2/tiny_attention.py - attention for rows of 12 places, as one small program of the graphics
card per row and head (written in Triton, the language torch.compile also writes in).

    from v2.tiny_attention import available, tiny_attention
    out = tiny_attention(qkv, heads)   # qkv [rows, time, 3 * columns] -> [rows, time, columns]

It computes exactly what v2/net.py's Attention computes between its two linear parts (causal
attention: a place looks at itself and the places before it), up to rounding. PyTorch's own
ways are made for long texts: for 12 places they waste most of the card's work, and the plain
matrix products read the same numbers many times. Here each program reads the query, key and
value of its row and head once, keeps all 12 x 12 scores in its registers and writes the result
once. The backward pass works the scores out again instead of keeping them.

Only on a graphics card with Triton; `available()` says whether it can be used.
"""

import math

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # no Triton (for example on the processor): the plain code is used instead
    triton = None

BLOCK_T = 16  # places, rounded up (the card's small matrix products need at least 16)


def available(device):
    return triton is not None and torch.device(device).type == "cuda"


if triton is not None:

    @triton.jit
    def _load(base, row_stride, offset, t, d, time, size):
        """A [BLOCK_T, BLOCK_D] block of one row: places t, columns offset + d (zeros outside)."""
        mask = (t[:, None] < time) & (d[None, :] < size)
        return tl.load(base + t[:, None] * row_stride + offset + d[None, :], mask=mask,
                       other=0.0)

    @triton.jit
    def _probabilities(q, k, t, time, scale):
        """softmax over the allowed places (j <= i) of the scores q k^T * scale."""
        scores = tl.dot(q, tl.trans(k), input_precision="ieee") * scale
        allowed = (t[None, :] <= t[:, None]) & (t[None, :] < time)
        scores = tl.where(allowed, scores, float("-inf"))
        top = tl.max(scores, axis=1)
        p = tl.exp(scores - top[:, None])
        return p / tl.sum(p, axis=1)[:, None]

    # Triton tries these ways of running the programs on the first call, and keeps the fastest
    # (for each number of columns and heads). They differ only in speed, not in the sums.
    WAYS = [triton.Config({}, num_warps=w, num_stages=s) for w in (1, 2, 4) for s in (1, 2)]

    @triton.autotune(configs=WAYS, key=["columns", "heads"])
    @triton.jit
    def _forward(qkv, out, time, columns, size, heads, scale,
                 BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr):
        row = tl.program_id(0) // heads
        head = tl.program_id(0) % heads
        t = tl.arange(0, BLOCK_T)
        d = tl.arange(0, BLOCK_D)
        base = qkv + row * time * 3 * columns
        q = _load(base, 3 * columns, head * size, t, d, time, size)
        k = _load(base, 3 * columns, columns + head * size, t, d, time, size)
        v = _load(base, 3 * columns, 2 * columns + head * size, t, d, time, size)
        p = _probabilities(q, k, t, time, scale)
        o = tl.dot(p, v, input_precision="ieee")
        mask = (t[:, None] < time) & (d[None, :] < size)
        tl.store(out + row * time * columns + t[:, None] * columns + head * size + d[None, :],
                 o, mask=mask)

    @triton.autotune(configs=WAYS, key=["columns", "heads"])
    @triton.jit
    def _backward(qkv, grad_out, grad_qkv, time, columns, size, heads, scale,
                  BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr):
        row = tl.program_id(0) // heads
        head = tl.program_id(0) % heads
        t = tl.arange(0, BLOCK_T)
        d = tl.arange(0, BLOCK_D)
        base = qkv + row * time * 3 * columns
        q = _load(base, 3 * columns, head * size, t, d, time, size)
        k = _load(base, 3 * columns, columns + head * size, t, d, time, size)
        v = _load(base, 3 * columns, 2 * columns + head * size, t, d, time, size)
        do = _load(grad_out + row * time * columns, columns, head * size, t, d, time, size)
        p = _probabilities(q, k, t, time, scale)
        dv = tl.dot(tl.trans(p), do, input_precision="ieee")
        dp = tl.dot(do, tl.trans(v), input_precision="ieee")
        ds = p * (dp - tl.sum(p * dp, axis=1)[:, None])
        dq = tl.dot(ds, k, input_precision="ieee") * scale
        dk = tl.dot(tl.trans(ds), q, input_precision="ieee") * scale
        mask = (t[:, None] < time) & (d[None, :] < size)
        place = grad_qkv + row * time * 3 * columns + t[:, None] * 3 * columns + d[None, :]
        tl.store(place + head * size, dq, mask=mask)
        tl.store(place + columns + head * size, dk, mask=mask)
        tl.store(place + 2 * columns + head * size, dv, mask=mask)


def _sizes(qkv, heads):
    rows, time, three_columns = qkv.shape
    columns = three_columns // 3
    size = columns // heads
    if time > BLOCK_T:
        raise ValueError(f"tiny_attention handles at most {BLOCK_T} places")
    return rows, time, columns, size, max(16, triton.next_power_of_2(size))


@torch.library.custom_op("air::tiny_attention", mutates_args=())
def tiny_attention(qkv: torch.Tensor, heads: int) -> torch.Tensor:
    """Causal attention of every row: qkv [rows, time, 3 * columns] (query, key and value side
    by side, each split into `heads` heads) -> [rows, time, columns]."""
    qkv = qkv.contiguous()
    rows, time, columns, size, block_d = _sizes(qkv, heads)
    out = torch.empty(rows, time, columns, device=qkv.device, dtype=qkv.dtype)
    _forward[(rows * heads,)](qkv, out, time, columns, size, heads, 1.0 / math.sqrt(size),
                              BLOCK_T=BLOCK_T, BLOCK_D=block_d)
    return out


@tiny_attention.register_fake
def _(qkv, heads):
    return qkv.new_empty(qkv.shape[0], qkv.shape[1], qkv.shape[2] // 3)


@torch.library.custom_op("air::tiny_attention_backward", mutates_args=())
def tiny_attention_backward(qkv: torch.Tensor, grad_out: torch.Tensor,
                            heads: int) -> torch.Tensor:
    qkv = qkv.contiguous()
    grad_out = grad_out.contiguous()
    rows, time, columns, size, block_d = _sizes(qkv, heads)
    grad_qkv = torch.empty_like(qkv)
    _backward[(rows * heads,)](qkv, grad_out, grad_qkv, time, columns, size, heads,
                               1.0 / math.sqrt(size), BLOCK_T=BLOCK_T, BLOCK_D=block_d)
    return grad_qkv


@tiny_attention_backward.register_fake
def _(qkv, grad_out, heads):
    return torch.empty_like(qkv)


def _setup(ctx, inputs, output):
    ctx.save_for_backward(inputs[0])
    ctx.heads = inputs[1]


def _grad(ctx, grad_out):
    (qkv,) = ctx.saved_tensors
    return tiny_attention_backward(qkv, grad_out, ctx.heads), None


tiny_attention.register_autograd(_grad, setup_context=_setup)
