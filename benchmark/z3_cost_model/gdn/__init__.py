"""FlashQLA GDN core benchmark port.

Only the dependency-free configuration symbols are exported here.  Import
``runner`` or ``correctness`` explicitly for GPU execution.
"""

from .config import (
    BLOCK_DV_DOMAIN,
    CORE_CONTRACT,
    GROUP_REDUCE_BLOCK_SIZE_DOMAIN,
    PREPARE_H_NUM_STAGES_DOMAIN,
    Direction,
    GDNConfig,
    default_config,
    raw_configs,
)

__all__ = [
    "BLOCK_DV_DOMAIN",
    "CORE_CONTRACT",
    "GROUP_REDUCE_BLOCK_SIZE_DOMAIN",
    "PREPARE_H_NUM_STAGES_DOMAIN",
    "Direction",
    "GDNConfig",
    "default_config",
    "raw_configs",
]
