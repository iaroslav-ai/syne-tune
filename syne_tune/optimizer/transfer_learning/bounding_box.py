import numpy as np
from typing import Dict, Callable

from blackbox_repository import load
import syne_tune.search_space as sp
from syne_tune.optimizer.scheduler import TrialScheduler
from syne_tune.optimizer.schedulers.fifo import FIFOScheduler
from syne_tune.optimizer.transfer_learning import TransferLearningScheduler, TransferLearningTaskEvaluations
from syne_tune.search_space import Categorical
from syne_tune.util import catchtime


class BoundingBox(TransferLearningScheduler):
    def __init__(
            self,
            scheduler_fun: Callable[[Dict, str, str], TrialScheduler],
            config_space: Dict,
            mode: str,
            metric: str,
            transfer_learning_evaluations: Dict[str, TransferLearningTaskEvaluations],
            num_hyperparameters: int = 30,
    ):
        self.mode = mode
        with catchtime("compute bounding box"):
            new_config_space = self.compute_box(config_space, transfer_learning_evaluations, mode=self.mode, min_choices=num_hyperparameters)
        # ugly
        self.config_space = new_config_space
        print(f"hyperparameter ranges of best previous config {new_config_space}")
        print(f"({sp.search_space_size(new_config_space)} options)")
        self.scheduler = scheduler_fun(new_config_space, mode, metric)
        super(BoundingBox, self).__init__(
            config_space=new_config_space,
            transfer_learning_evaluations=transfer_learning_evaluations,
            metric_names=self.scheduler.metric_names(),
        )

    @staticmethod
    def compute_box(config_space, transfer_learning_evaluations, mode: str, min_choices: int):
        # hps = []
        best_hp_indices = []
        for task, evaluation in transfer_learning_evaluations.items():
            best_hp_task_indices = evaluation.metrics.reshape(-1).argsort()
            if mode == 'max':
                # revert
                best_hp_task_indices = best_hp_task_indices[::-1]
            best_hp_indices.append(best_hp_task_indices)

        # (num_task, num_hyperparameters)
        best_hp_indices = np.stack(best_hp_indices)

        i = 1
        while i < best_hp_indices.shape[1] and len(set(best_hp_indices[:, :i].reshape(-1))) < min_choices:
            i += 1

        # at least min_choices
        hps = evaluation.hyperparameters.loc[set(best_hp_indices[:, :i].reshape(-1))].values

        # (num_best_hps, num_hyperparameters)
        hp_df = np.stack(hps).reshape(-1, hps[0].shape[-1])

        new_config_space = {}
        for i, (name, domain) in enumerate(config_space.items()):
            if isinstance(domain, Categorical):
                hp_values = list(sorted(set(hp_df[:, i])))
                new_config_space[name] = sp.choice(hp_values)
            else:
                # assume its numerical
                new_domain = domain
                new_domain.lower = hp_df[:, i].min()
                new_domain.upper = hp_df[:, i].max()
                new_config_space[name] = new_domain
        print(new_config_space)
        return new_config_space

    def suggest(self, *args, **kwargs):
        return self.scheduler.suggest(*args, **kwargs)

    def on_trial_add(self, *args, **kwargs):
        self.scheduler.on_trial_add(*args, **kwargs)

    def on_trial_complete(self, *args, **kwargs):
        self.scheduler.on_trial_complete(*args, **kwargs)

    def on_trial_remove(self, *args, **kwargs):
        self.scheduler.on_trial_remove(*args, **kwargs)

    def on_trial_error(self, *args, **kwargs):
        self.scheduler.on_trial_error(*args, **kwargs)

    def on_trial_result(self, *args, **kwargs) -> str:
        return self.scheduler.on_trial_result(*args, **kwargs)


if __name__ == '__main__':

    bb_dict = load("nasbench201")

    test_task = "cifar100"
    config_space = bb_dict[test_task].configuration_space
    metric_index = 0
    transfer_learning_evaluations = {
        task: TransferLearningTaskEvaluations(
            hyperparameters=bb.hyperparameters,
            # average over seed, take last fidelity and pick only first metric
            metrics=bb.objectives_evaluations.mean(axis=1)[:, -1, metric_index:metric_index+1]
        )
        for task, bb in bb_dict.items()
        if task != test_task
    }

    bb_sch = BoundingBox(
        scheduler_fun=lambda new_config_space, mode, metric: FIFOScheduler(
            new_config_space,
            points_to_evaluate=[],
            searcher='random',
            metric=metric,
            mode=mode,
        ),
        mode="min",
        config_space=config_space,
        metric=bb_dict[test_task].objectives_names[metric_index],
        transfer_learning_evaluations=transfer_learning_evaluations,
    )