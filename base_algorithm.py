# Implements the algorithm described by:
# Gao et al.
from algorithm import Algorithm, GridType
from environment import Highway
import math


class BaSEAlgorithm(Algorithm):
    def __init__(self, T: int, M: int, grid_type: GridType, environment: Highway, gamma: float) -> None:
        super().__init__(T, M, grid_type, environment)
        self._gamma = gamma

    def run(self) -> float:
        active_arms = set(self._environment.arms)
        K = len(active_arms)
        total_rewards = {x: 0 for x in active_arms}
        num_pulls = 0
        regret = 0
        best_arm = self._environment.best_arm
        for i in range(1, self._M):
            pull_amount = int(
                (self._grid[i] - self._grid[i-1])/len(active_arms))
            num_pulls += pull_amount
            avg_so_far = {}
            max_avg_so_far = -math.inf
            for arm in active_arms:
                reward_for_arm = 0
                for i in range(0, pull_amount):
                    pull_reward = self._environment.pull_arm(arm)
                    regret += best_arm[1] - pull_reward[0]
                    reward_for_arm += pull_reward[0]
                total_rewards[arm] += reward_for_arm
                arm_avg = total_rewards[arm] / num_pulls
                avg_so_far[arm] = arm_avg
                if arm_avg > max_avg_so_far:
                    max_avg_so_far = arm_avg
            # Pruning step
            prune_limit = math.sqrt(
                self._gamma * math.log(K*self._T)/num_pulls)
            active_arms = {arm for arm in active_arms if (
                max_avg_so_far - avg_so_far[arm]) < prune_limit}
        # Last batch. Commit to best arm
        final_avgs = [(s, total_rewards[s]/num_pulls) for s in active_arms]
        max_avg_reward = -math.inf
        selected_best_arm = ""
        for i in final_avgs:
            if i[1] > max_avg_reward:
                selected_best_arm = i[0]
        final_pulls = self._grid[self._M] - self._grid[self._M - 1]
        for i in range(0, final_pulls):
            pull_reward = self._environment.pull_arm(selected_best_arm)
            regret += best_arm[1] - pull_reward[0]
        return regret
