import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Set, Tuple, Union

import numpy as np
import torch

from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.core import CheckpointingException, maybe_load_config
from megatron.core.dist_checkpointing.dict_utils import (
    extract_matching_values,
    map_reduce,
    nested_values,
)
from megatron.core.dist_checkpointing.mapping import (
    ShardedBase,
    ShardedObject,
    ShardedStateDict,
    is_main_replica,
)
from megatron.core.dist_checkpointing.strategies.base import (
    LoadCommonStrategy,
    LoadShardedStrategy,
    SaveCommonStrategy,
    SaveShardedStrategy,
    StrategyAction,
    get_default_strategy,
)

logger = logging.getLogger(__name__)

# TODO
LocalMetadata = List[Union[ShardedTensor, ShardedObject]]
GlobalMetadata = List[LocalMetadata]


def verify_checkpoint_and_load_strategy(
    checkpoint_dir: str,
    sharded_strategy: Union[LoadShardedStrategy, Tuple[str, int], None] = None,
    common_strategy: Union[LoadCommonStrategy, Tuple[str, int], None] = None,
) -> Tuple[LoadShardedStrategy, LoadCommonStrategy]:
    """ Verifies if checkpoint metadata exists and matches given strategy.

    Args:
        checkpoint_dir (str): checkpoint directory
        sharded_strategy (LoadShardedStrategy, Tuple[str, int], optional): load strategy to be verified
            if compatible with the checkpoint content. If None, the default load strategy
            for the checkpoint backend will be returned.
    """
    if not Path(checkpoint_dir).exists():
        raise CheckpointingException(f'Checkpoint directory {checkpoint_dir} does not exist')

    saved_config = maybe_load_config(checkpoint_dir)
    if saved_config is None:
        raise CheckpointingException(f'{checkpoint_dir} is not a distributed checkpoint')

    if sharded_strategy is None:
        sharded_strategy = get_default_strategy(
            StrategyAction.LOAD_SHARDED,
            saved_config.sharded_backend,
            saved_config.sharded_backend_version,
        )
    elif isinstance(sharded_strategy, tuple):
        sharded_strategy = get_default_strategy(StrategyAction.LOAD_SHARDED, *sharded_strategy)

    if common_strategy is None:
        common_strategy = get_default_strategy(
            StrategyAction.LOAD_COMMON,
            saved_config.common_backend,
            saved_config.common_backend_version,
        )
    elif isinstance(common_strategy, tuple):
        sharded_strategy = get_default_strategy(StrategyAction.LOAD_COMMON, *common_strategy)

    # TODO: implement consistency checks here
    return sharded_strategy, common_strategy


def adjust_non_strict_load(
    sharded_state_dict: ShardedStateDict, unexpected_keys: Set[str],
):
    def should_remove_unexpected_keys(x: ShardedBase):
        assert isinstance(x, ShardedBase), f'Unexpected type {type(x)}'
        return x.key in unexpected_keys

    _, sharded_state_dict = extract_matching_values(
        sharded_state_dict, should_remove_unexpected_keys
    )
    return sharded_state_dict


def _determine_missing_and_unexpected_keys(
    ckpt_sharded_metadata: ShardedStateDict,
    local_metadata: LocalMetadata,
    global_metadata: GlobalMetadata,
) -> Tuple[Set[str], Set[str]]:
    """
    NOTE: asymmetry
    TODO
    Args:
        sharded_state_dict:
        ckpt_sharded_metadata:
        local_metadata:
        global_metadata:

    Returns:

    """
    global_accessed_keys = set(
        sh_base.key for rank_metadata in global_metadata for sh_base in rank_metadata
    )
    local_accessed_keys = set(sh_base.key for sh_base in local_metadata)
    ckpt_keys = set(sh_base.key for sh_base in ckpt_sharded_metadata.values())

    missing_keys = ckpt_keys - global_accessed_keys
    unexpected_keys = local_accessed_keys - ckpt_keys

    if missing_keys:
        logger.debug(f'Dist ckpt load missing keys: {missing_keys}')
    if unexpected_keys:
        logger.debug(f'Dist ckpt load unexpected keys: {unexpected_keys}')

    return missing_keys, unexpected_keys


def maybe_report_missing_and_unexpected_keys(
    missing_keys: Set[str], unexpected_keys: Set[str], raise_error: bool = True
) -> None:
    """
    TODO
    Args:
        missing_keys:
        unexpected_keys:
        raise_error:

    Returns:

    """
    if not missing_keys and not unexpected_keys:
        return
    missing_title_msg = (
        f'Some keys found in the checkpoint are missing in the provided sharded state dict. '
    )
    missing_body_msg = f'Missing keys (for all ranks): {missing_keys}. '
    unexpected_title_msg = f'Unexpected keys (not found in the checkpoint) encountered in the provided sharded state dict. '
    unexpected_body_msg = f'Unexpected keys (for this rank): {unexpected_keys}. '
    if missing_keys:
        _missing_msg = missing_title_msg + missing_body_msg
        if raise_error:
            _missing_msg += (
                ' NOTE: This warning will become an error in MCore v0.9.'
                ' Make sure to provide a sharded_state_dict covering the whole checkpoint,'
                ' or set `dist_checkpointing.load(..., strict=False)` flag'
            )
        logger.warning(_missing_msg)
    if unexpected_keys:
        _unexpected_msg = unexpected_title_msg + unexpected_body_msg
        if raise_error:
            raise CheckpointingException(_unexpected_msg)
        else:
            logger.warning(_unexpected_msg)


def validate_sharding_integrity(global_metadata: GlobalMetadata):
    """ Validate if the ShardedTensors from multiple processes define correct sharding of a global tensor.

    Local ShardedTensors metadata is exchanged with `torch.distributed.all_gather_object`
    and then process with global rank 0 checks if main replicas of the shards:
    - cover the whole global tensors
    - don't overlap

    Args:
        global_metadata (TODO): sharded tensors local to this process

    Returns:
        None

    Raises:
        CheckpointingException for invalid access pattern
    """
    if torch.distributed.get_rank() != 0:
        return

    key_shardings = defaultdict(list)
    for rank, rank_shardings in enumerate(global_metadata):
        for sharding in rank_shardings:
            key_shardings[sharding.key].append((rank, sharding))
    for key, shardings in key_shardings.items():
        if isinstance(shardings[0][1], ShardedObject):
            _validate_objects_for_key(shardings)
        else:
            _validate_sharding_for_key(shardings)


def _validate_sharding_for_key(rank_sharding: List[Tuple[int, ShardedTensor]]):
    some_rank_shard = rank_sharding[0][1]
    global_shape = some_rank_shard.global_shape
    local_shape = some_rank_shard.local_shape
    dtype = some_rank_shard.dtype
    has_flattened_range = some_rank_shard.flattened_range is not None
    for rank, sharding in rank_sharding:
        assert sharding.dtype == dtype, (sharding.dtype, dtype, some_rank_shard)
        assert sharding.global_shape == global_shape, (
            sharding.global_shape,
            global_shape,
            some_rank_shard,
        )
        assert sharding.local_shape == local_shape, (
            sharding.local_shape,
            local_shape,
            some_rank_shard,
        )
        assert (sharding.flattened_range is not None) == has_flattened_range, (
            (sharding.flattened_range is not None),
            has_flattened_range,
            some_rank_shard,
        )

    shard_access_cnt = _compute_shards_access(rank_sharding)
    if has_flattened_range:
        map_reduce(
            rank_sharding,
            lambda x: x[1].global_offset,
            lambda x: x[1],
            _validate_sharding_for_key_flattened,
        )
    else:
        if not torch.all(shard_access_cnt == 1):
            logger.error(f'Invalid access pattern for {rank_sharding[0][1]}: {shard_access_cnt}')
            raise CheckpointingException(f'Invalid access pattern for {rank_sharding[0][1]}')


def _compute_shards_access(rank_sharding):
    shard_access_cnt = torch.zeros(
        rank_sharding[0][1].axis_fragmentations, dtype=torch.int, device='cpu'
    )
    for rank, sharding in rank_sharding:
        if is_main_replica(sharding.replica_id):
            shard_access_cnt[sharding.local_chunk_offset_in_global()] += 1
        # TODO: consider validating different replicas too
    return shard_access_cnt


def _validate_sharding_for_key_flattened(tensors_by_shard):
    all_slices = []
    local_shape = tensors_by_shard[0].local_shape
    for sharding in tensors_by_shard:
        assert sharding.local_shape == local_shape
        sharding: ShardedTensor
        if not is_main_replica(sharding.replica_id):
            # TODO: this checks only saving (and loading replica_id=0) consistency
            continue

        all_slices.append((sharding.flattened_range.start, sharding.flattened_range.stop))

    starts, stops = map(np.asarray, zip(*sorted(all_slices)))
    if (
        starts[0] != 0
        or stops[-1] != np.product(local_shape)
        or not np.all(starts[1:] == stops[:-1])
    ):
        logger.error(
            f'Flattened ranges dont cover the whole shard {tensors_by_shard[0]}. Ranges: {(starts, stops)}'
        )
        raise CheckpointingException(
            f'Flattened ranges dont cover the whole shard {tensors_by_shard[0]}'
        )


def _validate_objects_for_key(sharded_objects: List[ShardedObject]):
    """ Ensure uniqueness of saved objects. """
    unique_keys = [
        sh_obj.unique_key for _, sh_obj in sharded_objects if is_main_replica(sh_obj.replica_id)
    ]
    if len(unique_keys) != len(set(unique_keys)):
        duplicates = {k: cnt for k, cnt in Counter(unique_keys).items() if cnt > 1}
        logger.error(f'Duplicate ShardedObject keys and counts: {duplicates}')
        raise CheckpointingException(f'Duplicate ShardedObject keys: {list(duplicates.keys())}')
    expected_shard_num = np.prod(sharded_objects[0][1].global_shape)
    if len(unique_keys) != expected_shard_num:
        err_msg = f'Invalid access pattern: {expected_shard_num - len(unique_keys)} ShardedObject are missing.'
        logger.error(f'{err_msg} Existing shards: {unique_keys}')
        raise CheckpointingException(err_msg)


def determine_global_metadata(
    sharded_state_dict: ShardedStateDict,
) -> Tuple[LocalMetadata, GlobalMetadata]:
    """
    TODO
    Args:
        sharded_state_dict:

    Returns:

    """
    local_metadata = [ten.without_data() for ten in nested_values(sharded_state_dict)]
    global_metadata = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(global_metadata, local_metadata)
    return local_metadata, global_metadata


def validate_sharded_objects_handling(
    sharded_strategy: Union[SaveShardedStrategy, LoadShardedStrategy],
    common_strategy: Union[SaveCommonStrategy, LoadCommonStrategy],
) -> None:
    """ Checks is either of the passed strategies can handle sharded objects.

    Args:
        sharded_strategy (Union[SaveShardedStrategy, LoadShardedStrategy]): sharded strategy used for saving/loading
        common_strategy (Union[SaveCommonStrategy, LoadCommonStrategy]): common strategy used for saving/loading

    Returns:
        None

    Raises:
        CheckpointingException: if both strategies can't handle ShardedObjects
    """
    if (
        not sharded_strategy.can_handle_sharded_objects
        and not common_strategy.can_handle_sharded_objects
    ):
        raise CheckpointingException(
            f'Either sharded strategy or common strategy must implement ShardedObjects handling.'
            f' Both {sharded_strategy} and {common_strategy} specify can_handle_sharded_objects=False'
        )
