import queue
from typing import List, Callable, Tuple
from torch.autograd import Variable
import torch
import torch.distributed as dist
import torch.nn.functional as F


TENSOR_SHAPES: List[Tuple[int]] = None
TENSOR_DTYPE: torch.dtype = None


def set_p2p_tensor_shapes(shapes: List[Tuple[int]]):
    global TENSOR_SHAPES
    TENSOR_SHAPES = shapes


def set_p2p_tensor_dtype(dtype: torch.dtype):
    global TENSOR_DTYPE
    TENSOR_DTYPE = dtype


def build_from_tensor_shapes():
    return [torch.empty(s, dtype=TENSOR_DTYPE, device="cpu", requires_grad=True) for s in TENSOR_SHAPES]


def append_irecv(ops: List[dist.P2POp], src: int, group: dist.ProcessGroup) -> List[torch.Tensor]:
    tensors = build_from_tensor_shapes()
    src = dist.distributed_c10d.get_global_rank(group, src)
    for tensor in tensors:
        if tensor is not None:
            ops.append(dist.P2POp(dist.irecv, tensor, src))
    return tensors


def append_isend(ops: List[dist.P2POp], tensors: List[torch.Tensor], dst: int, group: dist.ProcessGroup) -> None:
    dst = dist.distributed_c10d.get_global_rank(group, dst)
    for tensor in tensors:
        if tensor is not None:
            ops.append(dist.P2POp(dist.isend, tensor, dst))



def run_backward(tensors: List[torch.Tensor], grad_tensors: List[torch.Tensor]) -> None:
    kwargs = dict(
        keep_graph=False,
        create_graph=False,
        allow_unreachable=True,
        accumulate_grad=True,
    )
    Variable._execution_engine.run_backward(tensors, grad_tensors, **kwargs)


def chunk_tensor(x, chunks, dim):
    if x is None:
        return [None for _ in range(chunks)]
    return x.tensor_split(chunks, dim=dim)


def cat_tensor(x, dim):
    if (isinstance(x, tuple) or isinstance(x, list)):
        if len(x) == 1:
            return x[0]
        elif x[0] is None:
            assert all(y is None for y in x)
            return None
    return torch.cat(x, dim=dim)


def scatter(inputs, chunks, dim):
    assert isinstance(inputs, (torch.Tensor, tuple, list))
    if isinstance(inputs, torch.Tensor):
        inputs = (inputs,)
    assert all(x is None or isinstance(x, torch.Tensor) for x in inputs)
    inputs = [chunk_tensor(x, chunks, dim) for x in inputs]
    microbatches = [microbatch for microbatch in zip(*inputs)]
    if len(microbatches) == 0:
        microbatches = [() for _ in range(chunks)]
    return microbatches


def gather(micro_outputs, dim):
    assert isinstance(micro_outputs[0], (torch.Tensor, tuple, list))
    if isinstance(micro_outputs[0], torch.Tensor):
        micro_outputs = [(x,) for x in micro_outputs]
    outputs = [x for x in zip(*micro_outputs)]
    outputs = tuple(cat_tensor(x, dim=dim) for x in outputs)
    return outputs

def criterion(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(output, target).clone()


def ref_step(x, l, model, chunks):
    ys, losses = [], []
    for micro_x, micro_l in zip(x.chunk(chunks), l.chunk(chunks)):
        micro_y = model(micro_x)
        loss = criterion(micro_y, micro_l)
        loss.backward()
        ys.append(micro_y)
        losses.append(loss)
    y = torch.cat(ys, 0)
    loss = torch.stack(losses)
    return loss, y


def cal_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x.double(), y.double()
    cos_diff = 1 - 2 * (x * y).sum().item() / (x * x + y * y).sum().item()
    return cos_diff

