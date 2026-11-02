"""Share one TPU slice with the rollout engines, so weights reach them over ICI (--update-weight-transfer-mode ici).

Every trainer process gets MILES_TPU_SHARED_SLICE="<job slice-rank offset>,<slice size>,<store host>:<store port>"
(trainer offset 0) and every engine SGLANG_TPU_SHARED_SLICE for the same slice (its offset follows the trainer and
the engines before it), plus the slice-wide TORCH_TPU_SLICEBUILDER_ADDRESSES and TORCH_TPU_TOPOLOGY. libtpu builds
the slice from slice-wide RANK / WORLD_SIZE, while each job keeps its own torch.distributed world; that needs a
torch_tpu with job-local process groups (TORCH_TPU_JOB_DEVICE_IDS), so no job compiles programs over another's
devices. sglang_tpu/weight_sync/shared_slice.py is the engine side.
"""

import os
from dataclasses import dataclass
from datetime import timedelta

import torch.distributed as dist
from torch._C._distributed_c10d import Backend as C10dBackend

SHARED_SLICE_ENV = "MILES_TPU_SHARED_SLICE"
# Arbitrary; long enough for the other job to start its TPU runtime.
_JOIN_TIMEOUT = timedelta(minutes=30)
# Must match the engine's prefix: both jobs create the slice-wide backend on this store namespace.
_STORE_PREFIX = "tpu_shared_slice"


@dataclass(frozen=True)
class SharedSliceSpec:
    rank_offset: int
    size: int
    store_host: str
    store_port: int

    @classmethod
    def from_env(cls) -> "SharedSliceSpec | None":
        value = os.environ.get(SHARED_SLICE_ENV)
        if not value:
            return None
        offset, size, address = value.split(",")
        host, port = address.rsplit(":", 1)
        return cls(rank_offset=int(offset), size=int(size), store_host=host, store_port=int(port))


@dataclass(frozen=True)
class SharedSlice:
    """This process's place in the shared slice and the slice-wide backend that sends to the engine."""

    spec: SharedSliceSpec
    slice_rank: int
    backend: C10dBackend


def join(spec: SharedSliceSpec, *, job_rank: int, job_size: int) -> SharedSlice:
    """Join the slice and create its send/recv backend; must run before the process's first TPU op.

    torch_tpu fixes the job's devices (TORCH_TPU_JOB_DEVICE_IDS) at its first compile, and libtpu reads the
    slice-wide RANK / WORLD_SIZE when the runtime starts; the job's own values are restored afterwards.
    """
    # torch_tpu exists on TPU hosts only (the CPU driver reads SharedSliceSpec when it validates args), and
    # miles.platforms loads before the worker utilities, so neither is imported at module level.
    from torch_tpu._internal.distributed import tpu_distributed

    from miles.utils.workers.serving.utils import override_env

    addresses = os.environ.get("TORCH_TPU_SLICEBUILDER_ADDRESSES", "").split(",")
    if len(addresses) != spec.size:
        raise ValueError(
            f"{SHARED_SLICE_ENV} names a slice of {spec.size} devices, but TORCH_TPU_SLICEBUILDER_ADDRESSES lists "
            f"{addresses}: both jobs need the same slice-wide addresses"
        )
    if not os.environ.get("TORCH_TPU_TOPOLOGY"):
        raise ValueError(f"{SHARED_SLICE_ENV} needs TORCH_TPU_TOPOLOGY, the slice's topology, set on every job")
    slice_rank = spec.rank_offset + job_rank
    slice_env = {
        "RANK": str(slice_rank),
        "LOCAL_RANK": str(slice_rank),
        "WORLD_SIZE": str(spec.size),
        "LOCAL_WORLD_SIZE": str(spec.size),
        # libtpu only checks that these exist; the job's real ones are set before init_process_group
        "MASTER_ADDR": spec.store_host,
        "MASTER_PORT": str(spec.store_port),
    }
    with override_env(slice_env):
        device_id = tpu_distributed.global_device_id()  # starts the runtime, which joins the slice
        store = dist.TCPStore(
            spec.store_host, spec.store_port, spec.size, is_master=slice_rank == 0, timeout=_JOIN_TIMEOUT
        )
        store.set(f"device:{slice_rank}", str(device_id))
        os.environ["TORCH_TPU_JOB_DEVICE_IDS"] = ",".join(
            store.get(f"device:{spec.rank_offset + rank}").decode() for rank in range(job_size)
        )
        backend = tpu_distributed.create_process_group(
            dist.PrefixStore(_STORE_PREFIX, store), slice_rank, spec.size, _JOIN_TIMEOUT
        )
    return SharedSlice(spec=spec, slice_rank=slice_rank, backend=backend)
