"""
Chapter 2: Multi-Armed Bandit

Goal:
- Implement epsilon-greedy, decaying epsilon-greedy, UCB, and Thompson sampling.
- Compare their cumulative regret on the same Bernoulli bandit.

"""
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

class BernoulliBandit:
    """ K-armed Bernoulli Bandit. """
    def __init__(self, num_arms):
        self.probs = np.random.uniform(size=num_arms)
        self.best_idx = np.argmax(self.probs)
        self.best_prob = self.probs[self.best_idx]
        self.num_arms = num_arms

    def step(self, k):
        if np.random.rand() < self.probs[k]:
            return 1
        else:
            return 0
        
class Solver:
    """ Base class for bandit solvers. """
    def __init__(self, bandit):
        self.bandit = bandit
        self.counts = np.zeros(self.bandit.num_arms)
        self.regret = 0.0
        self.actions = []
        self.regrets = []

    def update_regret(self, arm):
        self.regret += self.bandit.best_prob - self.bandit.probs[arm]
        self.regrets.append(self.regret)

    def run_one_step(self):
        raise NotImplementedError
    
    def run(self, num_steps):
        for _ in range(num_steps):
            arm = self.run_one_step()
            self.counts[arm] += 1
            self.actions.append(arm)
            self.update_regret(arm)

class EpsilonGreedy(Solver):
    """ Choose a random arm with probability epsilon; otherwise choose the best estimate. """
    def __init__(self, bandit, epsilon= 0.01, init_prob=1.0):
        super(EpsilonGreedy, self).__init__(bandit)
        self.epsilon = epsilon
        self.estimates = np.array([init_prob] * self.bandit.num_arms)

    def run_one_step(self):
        if np.random.random() < self.epsilon:
            arm = np.random.randint(0, self.bandit.num_arms)
        else:
            arm = np.argmax(self.estimates)

        r = self.bandit.step(arm)
        self.estimates[arm] += 1.0 / (self.counts[arm] + 1) * (r - self.estimates[arm])
        return arm
    
class DecayingEpsilonGreedy(Solver):
    """ Use epsilon = 1 / t so exploration decrease over time. """
    def __init__(self, bandit, init_prob = 1.0):
        super(DecayingEpsilonGreedy, self).__init__(bandit)
        self.estimates = np.array([init_prob] * self.bandit.num_arms)
        self.total_count = 0

    def run_one_step(self):
        self.total_count += 1
        if np.random.random() < 1 / self.total_count:
            arm = np.random.randint(0, self.bandit.num_arms)
        else:
            arm = np.argmax(self.estimates)

        r = self.bandit.step(arm)
        self.estimates[arm] += 1.0 / (self.counts[arm] + 1) * (r - self.estimates[arm])

        return arm


def plot_results(solvers, solver_names, output_path):
    for solver, name in zip(solvers, solver_names):
        plt.plot(range(len(solver.regrets)), solver.regrets, label = name)

    plt.xlabel("Time steps")
    plt.ylabel("Cumulative regret")
    plt.title(f"{solvers[0].bandit.num_arms}-armed bandit")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def run_solver(solver_cls, bandit, num_steps, seed, *args, **kwargs):
    np.random.seed(seed)
    solver = solver_cls(bandit, *args, **kwargs)
    solver.run(num_steps)
    return solver

def main():
    num_arms = 10
    num_steps = 10000
    seed = 1

    np.random.seed(seed)
    bandit = BernoulliBandit(num_arms)
    print(f"随机生成了一个{num_arms}臂伯努利老虎机")
    print(f"获奖概率最大的拉杆为{bandit.best_idx}号, 其获奖概率为{bandit.best_prob:.4f}")

    solvers = [
        run_solver(EpsilonGreedy, bandit, num_steps, seed, epsilon=0.01),
        run_solver(DecayingEpsilonGreedy, bandit, num_steps, seed)
    ]
    solver_names = [
        "epsilon=0.01",
        "decaying epsilon"
    ]
    for name, solver in zip(solver_names, solvers):
        print(f"{name} 的累积懊悔为: {solver.regret:.4f}")

    output_path = Path(__file__).with_name('ch02_bandit_regret.png')
    plot_results(solvers, solver_names, output_path)
    print(f"对比图已保存到: {output_path}")

if __name__ == "__main__":
    main()