import argparse
import math
import statistics
import sys
import numpy as np
from typing import List, Tuple
from algorithm import GridType, Algorithm
from environment import Highway
import matplotlib.pyplot as plt


class AdaptiveAlgorithm(Algorithm):

    def __init__(self, T: int, M: int, environment: Highway) -> None:
        super().__init__(T=T, M=M, environment=environment, grid_type=GridType.adaptive)

    def run(self) -> Tuple[float, List[int]]:
        active_arms_over_batches = []
        active_arms = set(self._environment.arms)
        active_arms_over_batches.append(len(active_arms))
        K = len(active_arms)
        total_rewards = {x: 0 for x in active_arms}
        num_pulls = 0
        regret = 0
        best_arm = self._environment.best_arm
        remaining_time = self._T
        q = self._T**(1/self._M)
        for i in range(0, self._M):
            pull_amount = int(q ** i)
            if pull_amount * len(active_arms) > remaining_time:
                break
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
            remaining_time -= pull_amount * len(active_arms)
            # Prune
            prune_limit = math.sqrt(2*math.log(2*K*self._T*self._M)/num_pulls)
            active_arms = {arm for arm in active_arms if (
                max_avg_so_far - avg_so_far[arm]) <= prune_limit}
            active_arms_over_batches.append(len(active_arms))
        final_avgs = [(s, total_rewards[s]/num_pulls) for s in active_arms]
        max_avg_reward = -math.inf
        selected_best_arm = ""
        for i in final_avgs:
            if i[1] > max_avg_reward:
                selected_best_arm = i[0]
        if remaining_time > 0:
            pull_results = self._environment.pull_arm_n_times(
                selected_best_arm, remaining_time, True)
            reward_for_arm = np.sum(pull_results[0])
            regret += best_arm[1] * remaining_time - reward_for_arm
        return (regret, active_arms_over_batches)


def run(m: int, t: int, k: int = 3, iterations: int = 200) -> Tuple[float, List[float], List[float]]:
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
            algo = AdaptiveAlgorithm(
                t, m, highway)
            result = algo.run()
            regret += result[0]
            arms_data.append(result[1])
            print("Finish iteration={} m={}".format(i, m))
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


def generate_plots(m_min: int, m_max: int, t: int, k: int, iterations: int, prefix: str) -> None:
    m_values = [x for x in range(m_min, m_max+1)]
    regret = []
    for m in m_values:
        result = run(m, t, k, iterations)
        if result[0] == math.inf:
            break
        regret.append(result[0])
        plt.figure()
        plt.locator_params(axis='x', integer=True)

        plt.errorbar([x for x in range(len(result[1]))],
                     result[1], fmt='b', yerr=result[2])
        plt.xlabel('Batch Number')
        plt.ylabel('Number of active arms')
        title = '{}_arms_m_{}_t_{}_k_{}.png'.format(
            prefix, m, t, k)
        plt.savefig(title, bbox_inches='tight')
        plt.close()

    plt.figure()
    plt.locator_params(axis='x', integer=True)
    plt.plot(m_values[:len(regret)],
             regret, 'b')
    plt.xlabel('M')
    plt.ylabel('Average regret')
    title = '{}_regret_t_{}_k_{}.png'.format(prefix, t, k)
    plt.savefig(title, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the Adaptive algorithm with customizable inputs")
    parser.add_argument('--m_min', type=int,
                        help='Minimum number of batches', default=2)
    parser.add_argument('--m_max', type=int,
                        help='Maximum number of batches', default=8)
    parser.add_argument('-k', type=int, help='Number of arms', default=100)
    parser.add_argument(
        '-t', type=int, help='Total number of pulls', default=5*10**4)
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
    iterations = args.iterations
    generate_plots(m_min=m_min, m_max=m_max, t=t,
                   k=k, prefix=prefix, iterations=iterations)
