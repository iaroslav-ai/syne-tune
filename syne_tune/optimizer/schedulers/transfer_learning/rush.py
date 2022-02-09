# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
import pickle
from typing import Dict, List, Optional

import numpy as np

from syne_tune.backend.trial_status import Trial
from syne_tune.optimizer.scheduler import SchedulerDecision, TrialSuggestion
from syne_tune.optimizer.schedulers.hyperband import HyperbandScheduler
from syne_tune.optimizer.schedulers.transfer_learning import TransferLearningTaskEvaluations, TransferLearningScheduler


class RUSHScheduler(TransferLearningScheduler):
    def __init__(
            self,
            config_space: Dict,
            transfer_learning_evaluations: Dict[str, TransferLearningTaskEvaluations],
            metric: str,
            points_to_evaluate: Optional[List[Dict]] = None,
            **hyperband_kwargs
    ) -> None:
        """
        A transfer learning variation of Hyperband which uses previously well-performing hyperparameter configurations
        as an initialization. The best hyperparameter configuration of each individual task provided is evaluated.
        The one among them which performs best on the current task will serve as a hurdle and is used to prune
        other candidates. This changes the standard successive halving promotion as follows. As usual, only the top-
        performing fraction is promoted to the next rung level. However, these candidates need to be at least as good
        as the hurdle configuration to be promoted. In practice this means that much fewer candidates can be promoted.

        Reference: A resource-efficient method for repeated HPO and NAS.
        Giovanni Zappella, David Salinas, Cédric Archambeau. AutoML workshop @ ICML 2021.

        :param config_space: Configuration space for trial evaluation function.
        :param metric: objective name to optimize, must be present in transfer learning evaluations.
        :param transfer_learning_evaluations: dictionary from task name to offline evaluations.
        :param points_to_evaluate: when points_to_evaluate is not None, the provided configurations are evaluated first
        in addition to top performing configurations from other tasks and also serve to preemptively prune
        underperforming configurations
        """
        super().__init__(config_space=config_space,
                         transfer_learning_evaluations=transfer_learning_evaluations,
                         metric_names=[metric])
        threshold_candidates = RUSHScheduler._compute_best_hyperparameter_per_task(
            config_space=config_space,
            transfer_learning_evaluations=transfer_learning_evaluations,
            metric=metric,
            mode=hyperband_kwargs.get('mode', 'min')
        )
        if points_to_evaluate is not None:
            threshold_candidates += points_to_evaluate
            threshold_candidates = [dict(s) for s in set(frozenset(p.items()) for p in threshold_candidates)]

        self._hyperband_scheduler = HyperbandScheduler(config_space, metric=metric,
                                                       points_to_evaluate=threshold_candidates, **hyperband_kwargs)
        assert self._hyperband_scheduler.scheduler_type in ['promotion', 'stopping'],\
            "RUSH supports only type 'stopping' or 'promotion'"
        self._num_init_configs = len(threshold_candidates)
        self._thresholds = dict()  # thresholds at different resource levels that must be met

    @staticmethod
    def _compute_best_hyperparameter_per_task(config_space: Dict,
                                              transfer_learning_evaluations: Dict[str, TransferLearningTaskEvaluations],
                                              metric: str,
                                              mode: str) -> List[Dict]:
        argbest, best = (np.argmin, np.min) if mode == 'min' else (np.argmax, np.max)
        baseline_configurations = list()
        for evals in transfer_learning_evaluations.values():
            best_hpc_idx = argbest(best(evals.objective_values(objective_name=metric)[..., -1], axis=1))
            hpc = evals.hyperparameters.iloc[best_hpc_idx]
            baseline_configurations.append({
                key: hpc[key] for key in config_space
            })
        return baseline_configurations

    def on_trial_error(self, trial: Trial) -> None:
        self._hyperband_scheduler.on_trial_error(trial)

    def on_trial_result(self, trial: Trial, result: Dict) -> str:
        trial_decision = self._hyperband_scheduler.on_trial_result(trial, result)
        return self._on_trial_result(trial_decision, trial, result)

    def on_trial_remove(self, trial: Trial) -> None:
        self._hyperband_scheduler.on_trial_remove(trial)

    def on_trial_complete(self, trial: Trial, result: Dict) -> None:
        self._hyperband_scheduler.on_trial_complete(trial, result)

    def suggest(self, trial_id: int) -> Optional[TrialSuggestion]:
        return self._hyperband_scheduler.suggest(trial_id)

    def __getstate__(self) -> Dict:
        return {
            'hyperband_scheduler': pickle.dumps(self._hyperband_scheduler),
            'num_init_configs': pickle.dumps(self._num_init_configs),
            'thresholds': pickle.dumps(self._thresholds)
        }

    def __setstate__(self, state):
        self._hyperband_scheduler = pickle.loads(state['hyperband_scheduler'])
        self._num_init_configs = pickle.loads(state['num_init_configs'])
        self._thresholds = pickle.loads(state['thresholds'])

    def _on_trial_result(self, trial_decision: str, trial: Trial, result: Dict) -> str:
        if trial_decision == SchedulerDecision.STOP or \
                trial_decision == SchedulerDecision.PAUSE and self._hyperband_scheduler.scheduler_type == 'stopping':
            return trial_decision

        metric_val = float(result[self._hyperband_scheduler.metric])
        resource = int(result[self._hyperband_scheduler._resource_attr])
        trial_id = str(trial.trial_id)

        if self._is_milestone_reached(trial_decision, trial_id, resource):
            if self._is_in_points_to_evaluate(trial):
                self._thresholds[resource] = self._return_better(self._thresholds.get(resource),
                                                                 metric_val)
            elif not self._meets_threshold(metric_val, resource):
                return SchedulerDecision.STOP

        return trial_decision

    def _is_milestone_reached(self, trial_decision: str, trial_id: str, resource: int) -> bool:
        if self._hyperband_scheduler.scheduler_type == 'promotion':
            return trial_decision == SchedulerDecision.PAUSE
        else:
            rung_sys, bracket_id, skip_rungs = self._hyperband_scheduler.terminator._get_rung_system(trial_id)
            rungs = rung_sys._rungs[:(-skip_rungs if skip_rungs > 0 else None)]
            return resource in [rung.level for rung in rungs]

    def _is_in_points_to_evaluate(self, trial: Trial) -> bool:
        return int(trial.trial_id) < self._num_init_configs

    def _meets_threshold(self, metric_val: float, resource: int) -> bool:
        threshold = self._thresholds.get(resource)
        if threshold is None:
            return True
        if self.metric_mode() == 'min':
            return metric_val <= threshold
        else:
            return metric_val >= threshold

    def _return_better(self, val1: Optional[float], val2: Optional[float]) -> float:
        if self.metric_mode() == 'min':
            better_val = min(float('inf') if val1 is None else val1,
                             float('inf') if val2 is None else val2)
        else:
            better_val = max(float('-inf') if val1 is None else val1,
                             float('-inf') if val2 is None else val2)
        return better_val
