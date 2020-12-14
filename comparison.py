import math
from algorithm import GridType
import argparse
import sys
from algorithm import GridType
from base_algorithm import run as base_run
from adaptive_algorithm import run as adaptive_run
import matplotlib.pyplot as plt


def generate_plots(m_min: int, m_max: int, t: int, gamma: float, k: int, iterations: int, prefix: str) -> None:
    m_values = [x for x in range(m_min, m_max+1)]
    base_regret = []
    adaptive_regret = []
    for m in m_values:
        base_results = base_run(m, t, GridType.minimax, gamma, k, iterations)
        if base_results[0] == math.inf:
            break
        base_regret.append(base_results[0])
        adaptive_results = adaptive_run(m, t, k, iterations)
        if adaptive_results[0] == math.inf:
            break
        adaptive_regret.append(adaptive_results[0])
        plt.figure()
        plt.locator_params(axis='x', integer=True)
        plt.errorbar([x for x in range(len(base_results[1]))],
                     base_results[1], fmt='b', yerr=base_results[2], label='BaSE')
        plt.errorbar([x for x in range(len(adaptive_results[1]))],
                     adaptive_results[1], fmt='g', yerr=base_results[2], label='Adaptive')
        plt.xlabel('Batch Number')
        plt.ylabel('Number of active arms')
        plt.legend()
        title = '{}_arms_m_{}_t_{}_k_{}_gamma_{}.png'.format(
            prefix, m, t, k, gamma)
        plt.savefig(title, bbox_inches='tight')
        plt.close()
    plt.figure()
    plt.locator_params(axis='x', integer=True)
    plt.plot(m_values[:len(base_regret)], base_regret, 'b', label='BaSE')
    plt.plot(m_values[:len(adaptive_regret)],
             adaptive_regret, 'g', label='Adaptive')
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
    gamma = args.gamma
    iterations = args.iterations
    generate_plots(m_min=m_min, m_max=m_max, t=t,
                   gamma=gamma, k=k, prefix=prefix, iterations=iterations)
