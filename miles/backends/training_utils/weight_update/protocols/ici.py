"""Device-to-device weight sync over the ICI links of a TPU slice the trainer shares with its engines.

Every trainer rank holds the same fully gathered buckets, so the engine ranks are dealt round robin to the trainer
ranks, and each trainer rank sends each bucket, packed into one buffer, to its engine ranks; each engine rank views
the tensors back out of the buffer and loads them through its model's load_weights (sglang_tpu's "ici.receive"
handler, served by /plugin_weight_update). The slice comes from miles.platforms.tpu_shared_slice.
"""

from argparse import Namespace
from collections.abc import Callable, Iterator, Sequence

import torch
import torch.distributed as dist

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.training_utils.parallel import ParallelState
from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement
from miles.backends.training_utils.weight_update.protocol import WeightTransferProtocol
from miles.backends.training_utils.weight_update.rollout_cell_updater import create_rollout_cell_updaters
from miles.backends.training_utils.weight_update.session import check_weight_sync_results
from miles.platforms import current_platform
from miles.utils import async_utils
from miles.utils.distributed_utils import get_gloo_group

ICI_RECEIVE_HANDLER = "ici.receive"
# Many small sends crash libtpu's CopyToRemoteDevice, so a bucket travels as one buffer, each tensor in its own
# 4 KiB-aligned region. The engine allocates its receive buffer with the same dtype, which must match exactly.
_ALIGN_BYTES = 4096
_BUFFER_DTYPE = torch.int16


class UpdateWeightFromIci(WeightTransferProtocol):
    # every trainer rank sends whole tensors, so it must hold every PP stage's
    required_placement = WeightUpdatePlacement(gather_pp=True)

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self.group_name = "miles-ici"
        # the bucket's index in this sync; the engine receives with the tag the request names
        self._next_tag = 0

    def connect(
        self,
        rollout_engines: Sequence[SGLangApiClient],
        engine_gpu_counts: Sequence[int] | None,
        engine_gpu_offsets: Sequence[int] | None,
        engine_cell_ids: Sequence[str],
        parallel_state: ParallelState,
        placement: WeightUpdatePlacement,
        selector: str,
    ) -> None:
        shared_slice = current_platform.shared_slice
        if shared_slice is None:
            raise ValueError(
                "--update-weight-transfer-mode ici needs a TPU trainer started with MILES_TPU_SHARED_SLICE"
            )
        trainer_size = dist.get_world_size()
        engine_owners = assign_engine_ranks(trainer_size=trainer_size, engine_gpu_counts=engine_gpu_counts)
        if shared_slice.spec.size != trainer_size + sum(engine_gpu_counts) or list(engine_gpu_offsets) != [
            sum(engine_gpu_counts[:e]) for e in range(len(engine_gpu_counts))
        ]:
            raise ValueError(
                "--update-weight-transfer-mode ici needs the trainer and its engines to fill the shared slice, the "
                f"engines back to back after the trainer: trainer ranks {trainer_size}, engine devices "
                f"{engine_gpu_counts} at {engine_gpu_offsets}, slice size {shared_slice.spec.size}"
            )
        self.rollout_engines = rollout_engines
        self.cell_updaters_of_cell_id = create_rollout_cell_updaters(self.args, rollout_engines, engine_cell_ids)
        self.is_sender = True
        self._shared_slice = shared_slice
        trainer_offset = shared_slice.spec.rank_offset
        # per engine, the trainer slice rank each of its TP ranks receives from
        self._sources_of_cell_id = {
            cell_id: [trainer_offset + owner for owner in owners]
            for cell_id, owners in zip(engine_cell_ids, engine_owners, strict=True)
        }
        # per engine, the engine slice ranks this trainer rank sends to
        self._targets_of_cell_id = {
            cell_id: [
                trainer_offset + trainer_size + offset + tp_rank
                for tp_rank, owner in enumerate(owners)
                if owner == dist.get_rank()
            ]
            for cell_id, offset, owners in zip(engine_cell_ids, engine_gpu_offsets, engine_owners, strict=True)
        }

    def begin_sync(
        self,
        weight_version: int,
        iter_buckets: Callable[..., Iterator[list[tuple[str, torch.Tensor]]]],
    ) -> bool:
        self._next_tag = 0
        return True

    def send_bucket(self, bucket: list[tuple[str, torch.Tensor]]) -> None:
        buffer, entries = pack_bucket(bucket, device=current_platform.current_device())
        tag, self._next_tag = self._next_tag, self._next_tag + 1
        # This request posts the engine's receives; the send below blocks until they are posted, and the reply
        # comes once the data has landed, which also keeps rank 0's buffer alive until then.
        cell_ids = self._healthy_cell_ids()
        futures = []
        if dist.get_rank() == 0:
            payload = {"tag": tag, "num_bytes": buffer.numel() * buffer.element_size(), "tensors": entries}
            futures = [
                self.cell_updaters_of_cell_id[cell_id].submit_client_call(
                    "plugin_weight_update",
                    handler=ICI_RECEIVE_HANDLER,
                    payload={**payload, "src_slice_ranks": self._sources_of_cell_id[cell_id]},
                )
                for cell_id in cell_ids
            ]
        # one at a time: each send blocks until its receiver has posted
        for cell_id in cell_ids:
            for target in self._targets_of_cell_id[cell_id]:
                self._shared_slice.backend.send([buffer], target, tag).wait()
        replies = async_utils.wait_futures(futures)
        check_weight_sync_results(replies, is_lora=False)
        for reply in replies:
            # None: the request failed and the cell updater marked the cell errored
            if reply is not None:
                _check_every_rank_loaded(reply, num_tensors=len(entries))

    def _healthy_cell_ids(self) -> list[str]:
        """The cells that still take updates, as rank 0 sees them: a send to an errored cell would never be received."""
        cell_ids = [cell_id for cell_id, updater in self.cell_updaters_of_cell_id.items() if not updater.is_errored]
        shared = [cell_ids]
        dist.broadcast_object_list(shared, src=0, group=get_gloo_group())
        return shared[0]


def pack_bucket(bucket: list[tuple[str, torch.Tensor]], *, device: torch.device) -> tuple[torch.Tensor, list[list]]:
    """One flat buffer holding every tensor of ``bucket``, and its ``[name, dtype, shape, byte_offset]`` entries."""
    offsets, num_bytes = [], 0
    for _, tensor in bucket:
        offsets.append(num_bytes)
        num_bytes += _region_bytes(tensor.numel() * tensor.element_size())
    buffer = torch.empty(num_bytes // _BUFFER_DTYPE.itemsize, dtype=_BUFFER_DTYPE, device=device)
    for (_, tensor), offset in zip(bucket, offsets, strict=True):
        _region(buffer, byte_offset=offset, dtype=tensor.dtype, numel=tensor.numel()).copy_(tensor.reshape(-1))
    entries = [
        [name, str(tensor.dtype).removeprefix("torch."), list(tensor.shape), offset]
        for (name, tensor), offset in zip(bucket, offsets, strict=True)
    ]
    return buffer, entries


def _region_bytes(num_bytes: int) -> int:
    # an empty tensor still takes a region, so no two entries share an offset
    return max(_ALIGN_BYTES, -(-num_bytes // _ALIGN_BYTES) * _ALIGN_BYTES)


def _region(buffer: torch.Tensor, *, byte_offset: int, dtype: torch.dtype, numel: int) -> torch.Tensor:
    """``numel`` elements of ``dtype`` at ``byte_offset`` of the buffer, as a view."""
    start = byte_offset // buffer.element_size()
    length = -(-numel * dtype.itemsize // buffer.element_size())
    # [:numel] drops the padding byte of an odd-length 1-byte tensor
    return buffer[start : start + length].view(dtype)[:numel]


def _check_every_rank_loaded(reply: dict, *, num_tensors: int) -> None:
    if any(count != num_tensors for count in reply["results"]):
        raise RuntimeError(f"ICI weight sync: engine TP ranks loaded {reply['results']} of {num_tensors} tensors")


def assign_engine_ranks(*, trainer_size: int, engine_gpu_counts: Sequence[int]) -> list[list[int]]:
    """Per engine, the trainer rank that sends to each of its TP ranks: engine ranks in order, round robin."""
    owners, next_rank = [], 0
    for count in engine_gpu_counts:
        owners.append([(next_rank + tp_rank) % trainer_size for tp_rank in range(count)])
        next_rank += count
    return owners
