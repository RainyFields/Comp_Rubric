"""Out-of-tree verl 0.9.1 plugin: custom advantage estimators, V1 trainer subclass, multi-output agent-loop worker.

Importing this package registers the estimators (``foldgrpo``, ``compaction_gae``, ``compaction_grpo``, ``supo``) and the
``fold_sync`` trainer. Design: docs/plans/2026-09-24_verl091_port_design.md.
"""
from . import estimators  # noqa: F401
from .estimators import CUSTOM_ESTIMATORS, compute_custom_advantage  # noqa: F401
from .trainer import TRAINER_NAME, FoldSyncTrainer  # noqa: F401
