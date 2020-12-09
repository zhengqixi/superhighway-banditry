# Implements the algorithm described by:
# Gao et al.
from typing import Dict

from matplotlib.pyplot import grid
from algorithm import Algorithm, GridType
from environment import Highway
import math
import numpy as np
import matplotlib.pyplot as plt


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
            # This is to account for rounding issues
            self._grid[i] = len(active_arms) * pull_amount + self._grid[i-1]
            num_pulls += pull_amount
            avg_so_far = {}
            max_avg_so_far = -math.inf
            for arm in active_arms:
                pull_results = self._environment.pull_arm_n_times(
                    arm, pull_amount, True)
                reward_for_arm = np.sum(pull_results[0])
                regret += best_arm[1] * pull_amount - reward_for_arm
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
        for i in range(0, int(final_pulls)):
            pull_reward = self._environment.pull_arm(selected_best_arm)
            regret += best_arm[1] - pull_reward[0]
        return regret


def run(m: int, t: int, gridtype: GridType, gamma: float = 0.5, k: int = 3, iterations: int = 200) -> float:
    if iterations <= 0:
        raise Exception("Number of iterations must be more than 0")
    if k <= 1:
        raise Exception("Must have at least 1 arm")

    arms = [(str(x), 0.5, 1) for x in range(0, k)]
    arms[0] = ("0", 0.6, 1)
    highway = Highway.from_dict({
        "arms": arms,
        "nodes": []
    })
    regret = 0
    for i in range(0, iterations):
        algo = BaSEAlgorithm(
            t, m, gridtype, highway, gamma)
        regret += algo.run()
        print("Finish iteration={} m={} gridtype={}".format(i, m, gridtype))
    avg_regret = regret/iterations/t
    return avg_regret


if __name__ == "__main__":
    t = 5 * 10**4
    m_values = [x for x in range(2, 8)]
    iterations = 200
    arithmetic = [run(x, t, GridType.arithmetic, iterations=iterations)
                  for x in m_values]
    geometric = [run(x, t, GridType.geometric, iterations=iterations)
                 for x in m_values]
    minimax = [run(x, t, GridType.minimax, iterations=iterations)
               for x in m_values]
    plt.plot(m_values, arithmetic, 'b', label='arithmetic')
    plt.plot(m_values, geometric, label='geometric')
    plt.plot(m_values, minimax, label='minimax')
    plt.xlabel('M')
    plt.ylabel('Average regret')
    plt.legend()
    plt.savefig('BaSE_Result.png', bbox_inches='tight')
    plt.show()
    plt.close()
