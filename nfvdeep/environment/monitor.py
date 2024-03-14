from typing import List
from collections import defaultdict

import gym

import numpy as np
import numpy.typing as npt

from stable_baselines3.common.callbacks import BaseCallback

STATS = ["accepted", "rejected", "operating_servers"]
COSTS = ["cpu_cost", "memory_cost", "bandwidth_cost"]
UTILIZATIONS = ["cpu_utilization", "memory_utilization", "bandwidth_utilization"]

class StatsWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

        self.placements = {}
        self.statistics = defaultdict(float)

    def clear(self) -> None:
        self.placements = {}
        self.statistics = defaultdict(float)

    def step(
            self, 
            action: int
            ) -> tuple[npt.NDArray[any], float, bool, dict[str, bool]]:
        obs, reward, done, info = super().step(action)

        keys = [
            key for key in info if key in STATS + COSTS + UTILIZATIONS
        ]
        for key in keys:
            self.statistics[key] += info[key]

        self.statistics["ep_length"] += 1

        if "sc" in info:
            self.placements[info["sc"]] = info["placements"]

        return obs, reward, done, info


class EvalLogCallback(BaseCallback):
    def __init__(
            self,
            verbose: int = 0):
        
        super().__init__(verbose)

    def _on_step(self) -> None:
        eval_envs: List[StatsWrapper] = self.locals["callback"].eval_env.envs
        assert all([isinstance(env, StatsWrapper) for env in eval_envs])

        num_requests = [
            max(env.statistics["accepted"] + env.statistics["rejected"], 1)
            for env in eval_envs
        ]
        acceptance = [
            env.statistics["accepted"] / nreqs
            for env, nreqs in zip(eval_envs, num_requests)
        ]
        rejection = [
            env.statistics["rejected"] / nreqs
            for env, nreqs in zip(eval_envs, num_requests)
        ]

        self.logger.record("eval/acceptance_ratio", np.mean(acceptance))
        self.logger.record("eval/rejection_ratio", np.mean(rejection))

        costs = [
            {
                key: env.statistics[key] / env.statistics["ep_length"]
                for key in COSTS
            }
            for env in eval_envs
        ]
        costs = {key: np.mean([dic[key] for dic in costs]) for key in costs[0]}

        for key, value in costs.items():
            self.logger.record("eval/mean_{}".format(key), value)

        occupied = [
            {key: env.statistics[key] for key in UTILIZATIONS} for env in eval_envs
        ]
        occupied = {key: np.mean([dic[key] for dic in occupied]) for key in occupied[0]}

        for key, value in occupied.items():
            self.logger.record("eval/mean_{}".format(key), value)

        operating = np.mean(
            [
                env.statistics["operating_servers"] / env.statistics["ep_length"]
                for env in eval_envs
            ]
        )
        self.logger.record("eval/mean_operating_servers", operating)

        for env in eval_envs:
            env.clear()
