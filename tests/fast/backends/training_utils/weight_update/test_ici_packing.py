"""The packed ICI bucket layout that sglang_tpu's "ici.receive" handler views tensors back out of."""

import itertools

import torch

from miles.backends.training_utils.weight_update.protocols.ici import assign_engine_ranks, pack_bucket


def _bucket() -> list[tuple[str, torch.Tensor]]:
    generator = torch.Generator().manual_seed(0)
    return [
        ("odd_bf16", torch.randn(3, 5, generator=generator).to(torch.bfloat16)),
        ("transposed_fp32", torch.randn(4, 6, generator=generator).t()),
        ("odd_uint8", torch.randint(0, 255, (7,), generator=generator, dtype=torch.uint8)),
        ("empty", torch.empty(0, 4, dtype=torch.bfloat16)),
        ("past_one_region_int64", torch.arange(600, dtype=torch.int64)),
    ]


def test_every_tensor_reads_back_from_its_own_aligned_region():
    """Each entry's bytes start on a 4 KiB boundary, never overlap another's, and hold exactly that tensor."""
    bucket = _bucket()

    buffer, entries = pack_bucket(bucket, device=torch.device("cpu"))

    assert buffer.dim() == 1 and buffer.dtype == torch.int16
    raw = buffer.view(torch.uint8)
    regions = []
    for (name, tensor), (entry_name, dtype_name, shape, offset) in zip(bucket, entries, strict=True):
        dtype = getattr(torch, dtype_name)
        num_bytes = tensor.numel() * dtype.itemsize
        assert (entry_name, dtype, shape) == (name, tensor.dtype, list(tensor.shape))
        assert offset % 4096 == 0 and offset + num_bytes <= raw.numel()
        assert torch.equal(raw[offset : offset + num_bytes].view(dtype).reshape(shape), tensor)
        regions.append((offset, offset + max(num_bytes, 1)))
    regions.sort()
    assert all(end <= next_start for (_, end), (next_start, _) in itertools.pairwise(regions))


def test_engine_ranks_are_dealt_round_robin_to_the_trainer_ranks():
    """Every engine rank gets exactly one sender, and the senders share the engine ranks evenly."""
    # one engine filling the rest of the slice: trainer rank i feeds TP rank i
    assert assign_engine_ranks(trainer_size=4, engine_gpu_counts=[4]) == [[0, 1, 2, 3]]
    # three TP2 engines behind two trainer ranks: each trainer rank feeds one rank of every engine
    assert assign_engine_ranks(trainer_size=2, engine_gpu_counts=[2, 2, 2]) == [[0, 1], [0, 1], [0, 1]]
    # uneven: four TP1 engines behind four trainer ranks, then three TP1 engines behind two
    assert assign_engine_ranks(trainer_size=4, engine_gpu_counts=[1, 1, 1, 1]) == [[0], [1], [2], [3]]
    assert assign_engine_ranks(trainer_size=2, engine_gpu_counts=[1, 1, 1]) == [[0], [1], [0]]
    assert assign_engine_ranks(trainer_size=2, engine_gpu_counts=[]) == []
