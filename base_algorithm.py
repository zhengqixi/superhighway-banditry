# Implements the algorithm described by:
# Gao et al.
import statistics
from typing import List, Tuple
from algorithm import Algorithm, GridType
from environment import Highway
import math
import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse


class BaSEAlgorithm(Algorithm):
    def __init__(self, T: int, M: int, grid_type: GridType, environment: Highway, gamma: float) -> None:
        super().__init__(T, M, grid_type, environment)
        self._gamma = gamma

    def run(self) -> Tuple[float, List[int]]:
        active_arms_over_batches = []
        active_arms = set(self._environment.arms)
        active_arms_over_batches.append(len(active_arms))
        K = len(active_arms)
        total_rewards = {x: 0 for x in active_arms}
        num_pulls = 0
        regret = 0
        best_arm = self._environment.best_arm
        for i in range(1, self._M):
            print('grid i={}'.format(self._grid[i]))
            print('grid i-1={}'.format(self._grid[i-1]))
            print('active arms={}'.format(len(active_arms)))
            batch_pulls = self._grid[i] - self._grid[i-1]
            pull_amount = int(
                batch_pulls/len(active_arms))
            if pull_amount == 0:
                raise Exception(
                    "Must have at least 1 pull per active arm per batch")
            # This is to account for rounding issues
            leftover_pulls = int(batch_pulls) - pull_amount * len(active_arms)
            self._grid[i] = len(active_arms) * pull_amount + self._grid[i-1]
            num_pulls += pull_amount
            avg_so_far = {}
            max_avg_so_far = -math.inf
            max_arm_so_far = next(iter(active_arms))
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
                    max_arm_so_far = arm
            # For leftover pulls, just pull the best current arm
            # We do not factor this into the reward calculation
            pull_results = self._environment.pull_arm_n_times(
                max_arm_so_far, leftover_pulls, True)
            regret += best_arm[1] * leftover_pulls - np.sum(pull_results[0])
            # Pruning step
            prune_limit = math.sqrt(
                self._gamma * math.log(K*self._T)/num_pulls)
            active_arms = {arm for arm in active_arms if (
                max_avg_so_far - avg_so_far[arm]) < prune_limit}
            active_arms_over_batches.append(len(active_arms))
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
        return (regret, active_arms_over_batches)


def run(m: int, t: int, gridtype: GridType, gamma: float = 0.5, k: int = 3, iterations: int = 200) -> Tuple[float, List[float], List[float]]:
    if iterations <= 0:
        raise Exception("Number of iterations must be more than 0")
    if k <= 1:
        raise Exception("Must have at least 1 arm")

    regret = 0
    arms_data = []
    try:
        for i in range(iterations):
            arm_values = np.random.default_rng().normal(loc=1, size=k)

            arms = [(str(x), float(x), 1) for x in arm_values]
            highway = Highway.from_dict({
                "arms": arms,
                "nodes": []
            })
            algo = BaSEAlgorithm(
                t, m, gridtype, highway, gamma)
            result = algo.run()
            regret += result[0]
            arms_data.append(result[1])
            print("Finish iteration={} m={} gridtype={}".format(i, m, gridtype))
    except Exception as e:
        print("Well that's a shame: {}".format(e))
        return (math.inf, [], [])
    avg_regret = regret/iterations/t
    avg_arm_per_batch = []
    stddev_arm_per_batch = []
    for i in range(m):
        column = [x[i] for x in arms_data]
        avg = statistics.mean(column)
        stddev = statistics.stdev(column)
        avg_arm_per_batch.append(avg)
        stddev_arm_per_batch.append(stddev)
    return (avg_regret, avg_arm_per_batch, stddev_arm_per_batch)


def generate_plots(m_min: int, m_max: int, t: int, gamma: float, k: int, iterations: int, prefix: str) -> None:
    m_values = [x for x in range(m_min, m_max+1)]
    arithmetic_skip = False
    geometric_skip = False
    minimax_skip = False
    arithmetic_regret = []
    geometric_regret = []
    minimax_regret = []
    for m in m_values:
        arithmetic_arms = None
        geometric_arms = None
        minimax_arms = None
        if not arithmetic_skip:
            result = run(m, t, GridType.arithmetic, gamma, k, iterations)
            if result[0] == math.inf:
                arithmetic_skip = True
            else:
                arithmetic_regret.append(result[0])
                arithmetic_arms = (result[1], result[2])
        if not geometric_skip:
            result = run(m, t, GridType.geometric, gamma, k, iterations)
            if result[0] == math.inf:
                geometric_skip = True
            else:
                geometric_regret.append(result[0])
                geometric_arms = (result[1], result[2])
        if not minimax_skip:
            result = run(m, t, GridType.minimax, gamma, k, iterations)
            if result[0] == math.inf:
                minimax_skip = True
            else:
                minimax_regret.append(result[0])
                minimax_arms = (result[1], result[2])
        plt.figure()
        plt.locator_params(axis='x', integer=True)
        if arithmetic_arms is not None:
            plt.errorbar([x for x in range(len(arithmetic_arms[0]))],
                         arithmetic_arms[0], fmt='b', yerr=arithmetic_arms[1], label='arithmetic')
        if geometric_arms is not None:
            plt.errorbar([x for x in range(len(geometric_arms[0]))],
                         geometric_arms[0], fmt='g', yerr=geometric_arms[1], label='geometric')
        if minimax_arms is not None:
            plt.errorbar([x for x in range(len(minimax_arms[0]))],
                         minimax_arms[0], fmt='r', yerr=minimax_arms[1], label='minimax')
        plt.xlabel('Batch Number')
        plt.ylabel('Number of active arms')
        plt.legend()
        title = '{}_arms_m_{}_t_{}_k_{}_gamma_{}.png'.format(
            prefix, m, t, k, gamma)
        plt.savefig(title, bbox_inches='tight')
        plt.close()

    plt.figure()
    plt.locator_params(axis='x', integer=True)
    plt.plot(m_values[:len(arithmetic_regret)],
             arithmetic_regret, 'b', label='arithmetic')
    plt.plot(m_values[:len(geometric_regret)],
             geometric_regret, 'g', label='geometric')
    plt.plot(m_values[:len(minimax_regret)],
             minimax_regret, 'r', label='minimax')
    plt.xlabel('M')
    plt.ylabel('Average regret')
    plt.legend()
    title = '{}_regret_t_{}_k_{}_gamma_{}.png'.format(prefix, t, k, gamma)
    plt.savefig(title, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the BaSE algorithm with customizable inputs")
    parser.add_argument('-g', '--gamma', type=float,
                        help='gamma variable for BaSE algorithm', default=0.5)
    parser.add_argument('--m_min', type=int,
                        help='Minimum number of batches', default=2)
    parser.add_argument('--m_max', type=int,
                        help='Maximum number of batches', default=8)
    parser.add_argument('-k', type=int, help='Number of arms', default=100)
    parser.add_argument(
        '-t', type=int, help='Total number of pulls', default=5*10**6)
    parser.add_argument(
        '-i', '--iterations', type=int, help='Iterations per trial', default=200
    )
    parser.add_argument('prefix', help='Savefile prefix')
    args = parser.parse_args()
    if args.m_min > args.m_max:
        print("Max batch size must be greater than min batch size")
        sys.exit(-1)
    if args.m_min < 2:
        print("Must have at least 2 batches")
        sys.exit(-1)
    prefix = args.prefix
    t = args.t
    m_max = args.m_max
    m_min = args.m_min
    k = args.k
    gamma = args.gamma
    iterations = args.iterations
    generate_plots(m_min=m_min, m_max=m_max, t=t,
                   gamma=gamma, k=k, prefix=prefix, iterations=iterations)
