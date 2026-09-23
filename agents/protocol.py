"""Explicit training–inference protocol settings.

Checkpoints trained before commit 3f697bf (the Qwen3-8B campaigns of Aug–Sep 2026: norl/grpo/foldgrpo/compactiongrpo) were rolled
out with a harness that differs from the current one in five places. Evaluating such a checkpoint with the new harness changes the
token stream it sees relative to training (measured: fewer branches, more window exhaustion). New training runs must use the
corrected protocol. Both are supported explicitly; nothing is switched silently.

    plugin.protocol = "v2"      (default) corrected protocol, use for every new training run and its evaluations
    plugin.protocol = "legacy"  reproduce the pre-3f697bf training-time rollout format for old checkpoints

Each switch can also be set individually (``plugin.<switch>``) and overrides the protocol default:

| switch                | legacy                                   | v2                                                     |
|-----------------------|------------------------------------------|--------------------------------------------------------|
| turn_end_newline      | False: sampled turns end at <|im_end|>   | True: <|im_end|>\\n (template-canonical), newline untrained |
| enforce_turn_cap      | False: plugin.turn_max_new_tokens ignored | True: cap sent to the engine                          |
| branch_inherit        | "render": branch re-tokenises history    | "ids": branch inherits the parent's exact token ids    |
| tail_inherit          | "render": compaction tail re-rendered    | "ids": tail carried as the previous segment's raw ids  |
| action_grammar        | "legacy": env = line-anchored closed calls on the RAW text, last adjacent group (<4 newlines), calls inside <think> can execute; agent = lenient last-closed-call for branch, substring `<function=return>` ends a branch | "strict": think blocks stripped first, every closed line-anchored call executes in order (≤ max_calls_per_turn), finish/return/branch must be alone; a malformed (unclosed) `return` ends a branch as a flagged malformed return |

Independent of the protocol (safety nets, both modes): max_consecutive_no_call (3), max_calls_per_turn (8; 0 = off).
"""
from dataclasses import dataclass

DEFAULTS = {
    "legacy": dict(turn_end_newline=False, enforce_turn_cap=False, branch_inherit="render", tail_inherit="render", action_grammar="legacy"),
    "v2": dict(turn_end_newline=True, enforce_turn_cap=True, branch_inherit="ids", tail_inherit="ids", action_grammar="strict"),
}


@dataclass(frozen=True)
class Protocol:
    name: str
    turn_end_newline: bool
    enforce_turn_cap: bool
    branch_inherit: str
    tail_inherit: str
    action_grammar: str

    @property
    def legacy_grammar(self) -> bool:
        return self.action_grammar == "legacy"


def resolve(plugin) -> Protocol:
    """Read ``plugin.protocol`` (+ per-switch overrides) from a rollout config's plugin namespace (or None)."""
    g = (lambda k, d: getattr(plugin, k, d)) if plugin is not None else (lambda k, d: d)
    name = str(g("protocol", "v2")).lower()
    if name not in DEFAULTS:
        raise ValueError(f"plugin.protocol must be one of {list(DEFAULTS)}, got {name!r}")
    d = dict(DEFAULTS[name])
    for k in d:
        v = g(k, None)
        if v is not None:
            d[k] = (str(v).lower() in ("1", "true", "yes")) if isinstance(d[k], bool) else str(v)
    return Protocol(name=name, **d)


def resolve_from_config(config) -> Protocol:
    """``config`` is a rollout config (has ``.plugin``) or a trainer config (``actor_rollout_ref.rollout.plugin``)."""
    plugin = getattr(config, "plugin", None)
    if plugin is None and hasattr(config, "actor_rollout_ref"):
        plugin = getattr(config.actor_rollout_ref.rollout, "plugin", None)
    return resolve(plugin)
