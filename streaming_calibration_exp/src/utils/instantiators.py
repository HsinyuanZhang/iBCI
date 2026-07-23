"""Hydra instantiation helpers."""
from typing import List

import hydra
from lightning import Callback
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def _collect_target_nodes(node, found: List[DictConfig]) -> None:
    """Collect leaf DictConfigs that declare _target_.

    If a node has both _target_ and nested DictConfig children (Hydra merge
    artifact), recurse into children only and ignore the polluted root target.
    """
    if not isinstance(node, DictConfig):
        return

    child_dicts: List[DictConfig] = []
    for key in node.keys():
        if key == "_target_":
            continue
        child = node._get_child(key)
        if isinstance(child, DictConfig):
            child_dicts.append(child)
    if "_target_" in node and not child_dicts:
        found.append(node)
        return

    for child in child_dicts:
        _collect_target_nodes(child, found)


def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    callbacks: List[Callback] = []
    if not callbacks_cfg:
        log.warning("No callback configs found! Skipping..")
        return callbacks
    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    nodes: List[DictConfig] = []
    _collect_target_nodes(callbacks_cfg, nodes)
    for cb_conf in nodes:
        log.info(f"Instantiating callback <{cb_conf._target_}>")
        callbacks.append(hydra.utils.instantiate(cb_conf))
    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> List[Logger]:
    logger: List[Logger] = []
    if not logger_cfg:
        log.warning("No logger configs found! Skipping...")
        return logger
    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    nodes: List[DictConfig] = []
    _collect_target_nodes(logger_cfg, nodes)
    for lg_conf in nodes:
        log.info(f"Instantiating logger <{lg_conf._target_}>")
        logger.append(hydra.utils.instantiate(lg_conf))
    return logger
