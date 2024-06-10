# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

""" Common strategies. """

import logging
import os
from pathlib import Path

import torch

from megatron.core.dist_checkpointing.mapping import StateDict, ShardedStateDict
from megatron.core.dist_checkpointing.strategies.base import SaveCommonStrategy, \
    StrategyAction, default_strategies
from ..dict_utils import (
    dict_list_map_inplace,
    nested_values, )
from ..mapping import (
    CheckpointingException,
    ShardedObject,
    is_main_replica, )
from ..strategies.base import (
    LoadCommonStrategy,
)

_import_trigger = None

COMMON_STATE_FNAME = 'common.pt'

logger = logging.getLogger(__name__)


class TorchCommonSaveStrategy(SaveCommonStrategy):

    def save_common(self, common_state_dict: StateDict, checkpoint_dir: Path):
        if torch.distributed.get_rank() == 0:
            torch.save(common_state_dict, checkpoint_dir / COMMON_STATE_FNAME)

    def save_sharded_objects(self, sharded_objects_state_dict: ShardedStateDict,
                             checkpoint_dir: Path):

        for sh_obj in nested_values(sharded_objects_state_dict):
            if is_main_replica(sh_obj.replica_id):
                save_path = (checkpoint_dir / sh_obj.unique_key).with_suffix('.pt')
                os.makedirs(save_path.parent, exist_ok=True)
                torch.save(sh_obj.data, save_path)

    def can_handle_sharded_objects(self):
        return True


class TorchCommonLoadStrategy(LoadCommonStrategy):

    def load_common(self, checkpoint_dir: Path):
        """ Load common (non-sharded) objects state dict from the checkpoint.

        Args:
            checkpoint_dir (Path): checkpoint directory

        Returns:
            StateDict: state dict with non-sharded objects from the checkpoint
        """
        load_path = Path(checkpoint_dir) / COMMON_STATE_FNAME
        try:
            return torch.load(load_path, map_location='cpu')
        except FileNotFoundError as e:
            err_msg = f'Common file {load_path} does not exist'
            ckpt_files = [f.name for f in checkpoint_dir.iterdir()]
            logger.debug(f'{err_msg}. Checkpoint directory content: {ckpt_files}')
            raise CheckpointingException(err_msg) from e

    def load_sharded_objects(self, sharded_objects_state_dict: ShardedStateDict,
                             checkpoint_dir: Path):
        """ Replaces all ShardedObject from a given state dict with values loaded from the checkpoint.

        Args:
            sharded_objects_state_dict (ShardedStateDict): sharded state dict defining what objects should be loaded.
            checkpoint_dir (Path): checkpoint directory

        Returns:
            None: sharded state dict is modified in place
        """

        def load_sharded_object(sh_obj: ShardedObject):
            sh_obj.data = None
            load_path = (checkpoint_dir / sh_obj.unique_key).with_suffix('.pt')
            try:
                loaded_obj = torch.load(load_path)
            except FileNotFoundError as e:
                err_msg = f'Object shard {load_path} not found'
                obj_subdir = checkpoint_dir / sh_obj.key
                if obj_subdir.exists():
                    obj_files = [f.name for f in obj_subdir.iterdir()]
                    logger.debug(f'{err_msg}. Object {sh_obj.key} directory content: {obj_files}')
                else:
                    ckpt_files = [f.name for f in checkpoint_dir.iterdir()]
                    logger.debug(
                        f'{err_msg}. Object {sh_obj.key} directory does not exist. Checkpoint directory content: {ckpt_files}'
                    )
                raise CheckpointingException(err_msg) from e
            return loaded_obj

        return dict_list_map_inplace(load_sharded_object, sharded_objects_state_dict)

    @property
    def can_handle_sharded_objects(self):
        return True

    def check_backend_compatibility(self, loaded_version):
        pass

    def check_version_compatibility(self, loaded_version):
        pass


default_strategies[StrategyAction.LOAD_COMMON.value][('torch', 1)] = TorchCommonLoadStrategy()
default_strategies[StrategyAction.SAVE_COMMON.value][('torch', 1)] = TorchCommonSaveStrategy('torch', 1)
