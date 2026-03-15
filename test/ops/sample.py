import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

import llaisys
import torch

from test_utils import check_equal, llaisys_device, torch_device


def tensor_from_torch(torch_tensor, device_name):
    out = llaisys.Tensor(
        torch_tensor.shape,
        dtype={
            torch.float32: llaisys.DataType.F32,
            torch.float16: llaisys.DataType.F16,
            torch.bfloat16: llaisys.DataType.BF16,
            torch.int64: llaisys.DataType.I64,
        }[torch_tensor.dtype],
        device=llaisys_device(device_name),
    )
    api = llaisys.RuntimeAPI(llaisys_device(device_name))
    api.memcpy_sync(
        out.data_ptr(),
        torch_tensor.data_ptr(),
        torch_tensor.numel() * torch_tensor.element_size(),
        llaisys.MemcpyKind.D2D,
    )
    return out


def test_topk1_matches_argmax(device_name="cpu"):
    vals = torch.tensor([0.1, 2.0, -1.0, 1.5], dtype=torch.float32, device=torch_device(device_name))
    vals_ = tensor_from_torch(vals, device_name)
    idx = torch.tensor([1], dtype=torch.int64, device=torch_device(device_name))
    idx_ = tensor_from_torch(torch.zeros((1,), dtype=torch.int64, device=torch_device(device_name)), device_name)
    llaisys.Ops.sample(idx_, vals_, temperature=1.0, top_k=1, top_p=1.0, seed=1234)
    assert check_equal(idx_, idx, strict=True)


def test_seed_reproducible(device_name="cpu"):
    vals = torch.tensor([3.0, 2.0, 1.0, 0.0], dtype=torch.float32, device=torch_device(device_name))
    vals_ = tensor_from_torch(vals, device_name)
    idx1_ = tensor_from_torch(torch.zeros((1,), dtype=torch.int64, device=torch_device(device_name)), device_name)
    idx2_ = tensor_from_torch(torch.zeros((1,), dtype=torch.int64, device=torch_device(device_name)), device_name)
    llaisys.Ops.sample(idx1_, vals_, temperature=0.9, top_k=3, top_p=0.95, seed=777)
    llaisys.Ops.sample(idx2_, vals_, temperature=0.9, top_k=3, top_p=0.95, seed=777)
    assert check_equal(idx1_, tensor_to_torch(idx2_, device_name), strict=True)


def tensor_to_torch(tensor, device_name):
    torch_tensor = torch.zeros(tensor.shape(), dtype=torch.int64, device=torch_device(device_name))
    api = llaisys.RuntimeAPI(llaisys_device(device_name))
    api.memcpy_sync(
        torch_tensor.data_ptr(),
        tensor.data_ptr(),
        torch_tensor.numel() * torch_tensor.element_size(),
        llaisys.MemcpyKind.D2D,
    )
    return torch_tensor


def test_topp_limits_candidates(device_name="cpu"):
    vals = torch.tensor([5.0, 4.0, -10.0, -10.0], dtype=torch.float32, device=torch_device(device_name))
    vals_ = tensor_from_torch(vals, device_name)
    idx_ = tensor_from_torch(torch.zeros((1,), dtype=torch.int64, device=torch_device(device_name)), device_name)
    llaisys.Ops.sample(idx_, vals_, temperature=1.0, top_k=0, top_p=0.6, seed=42)
    result = int(tensor_to_torch(idx_, device_name)[0].item())
    assert result in (0, 1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
    args = parser.parse_args()

    print(f"Testing Ops.sample on {args.device}")
    test_topk1_matches_argmax(args.device)
    test_seed_reproducible(args.device)
    test_topp_limits_candidates(args.device)
    print("\033[92mTest passed!\033[0m\n")
