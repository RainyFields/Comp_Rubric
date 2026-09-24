"""Launcher: verl 0.9.1 ``main_ppo`` with the plugin registered inside the Ray TaskRunner process.

``get_trainer_cls(config.trainer.v1.trainer_mode)`` runs inside the TaskRunner actor, so the trainer class (and the
estimators) must be imported there, not only in the driver. ``FoldTaskRunner.run`` imports the plugin first.
Agent loops are imported by the workers through ``agent_loop_config_path`` (hydra ``_target_``), as before.
"""
import os

import hydra
import ray

import verl.trainer
from verl.trainer.main_ppo import TaskRunnerV1, run_ppo

_RunnerBase = TaskRunnerV1.__ray_metadata__.modified_class.__ray_actor_class__


@ray.remote
class FoldTaskRunner(_RunnerBase):
    def run(self, config):
        import agents.verl_plugin  # noqa: F401  (registers estimators + FoldSyncTrainer in this process)
        return super().run(config)


@hydra.main(config_path=os.path.join(os.path.dirname(verl.trainer.__file__), "config"), config_name="ppo_trainer",
            version_base=None)
def main(config):
    if not config.trainer.get("use_v1", True):
        raise ValueError("Comp_Rubric requires trainer.use_v1=True (multi-output rollouts need the V1 trainer)")
    run_ppo(config, task_runner_class=FoldTaskRunner)


if __name__ == "__main__":
    main()
