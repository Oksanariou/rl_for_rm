# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

from q_learning import q_to_v
from visualization_and_metrics import visualize_policy_RM, average_n_episodes, visualisation_value_RM, q_to_policy_RM

class Callback(object):
    def __init__(self, condition, agent):
        super(Callback).__init__()
        self.condition = condition
        self.agent = agent
        self.env = agent.env

    def run(self, episode):
        if not self.condition(episode):
            return
        self._run()

    def _run(self):
        raise NotImplementedError


class AgentMonitor(Callback):

    def _run(self):
        print("episode: {}, replay count: {}, loss: {:.2}, e: {:.2}, b: {:.2}".format(
            self.agent.episode, self.agent.replay_count, self.agent.loss_value, self.agent.epsilon,
            self.agent.priority_b))


class TrueCompute(Callback):

    def __init__(self, condition, agent):
        super().__init__(condition, agent)
        self.Q_table = None
        self.policy = None
        self.V_table = None

        self.name = "true_compute"

    def _run(self):
        true_Q_table, true_policy = self.agent.get_true_Q_table()
        true_V = q_to_v(self.env, true_Q_table)

        self.Q_table = true_Q_table
        self.policy = true_policy
        self.V_table = true_V


class QCompute(Callback):

    def __init__(self, condition, agent):
        super().__init__(condition, agent)
        self.Q_table = None
        self.policy = None
        self.V_table = None

        self.Q_tables = []
        self.policies = []
        self.V_tables = []

        self.name = "q_compute"

    def _run(self):
        Q_table = self.agent.compute_q_table()
        policy = q_to_policy_RM(self.env, Q_table)
        V = q_to_v(self.env, Q_table)

        self.Q_table = Q_table
        self.policy = policy
        self.V_table = V

        self.Q_tables.append(Q_table)
        self.policies.append(policy)
        self.V_tables.append(V)

    def reset(self, agent):
        self.agent = agent

        self.Q_table = None
        self.policy = None
        self.V_table = None

        self.Q_tables = []
        self.policies = []
        self.V_tables = []



class QErrorMonitor(Callback):

    def __init__(self, condition, agent, true_compute, q_compute):
        super().__init__(condition, agent)
        self.replays = []
        self.errors_Q_table = []
        self.errors_V_table = []
        self.errors_policy = []

        self.true = true_compute
        self.q = q_compute

        self.name = "q_error"

        borders = np.reshape(np.arange(self.env.T * self.env.C * self.env.nA), (self.env.T, self.env.C, self.env.nA))
        borders = ((borders // self.env.nA + 1) % 4 == 0) | (borders >= (self.env.T - 1) * self.env.C * self.env.nA)
        self.mask_borders = borders.reshape((self.env.T * self.env.C, self.env.nA))
        self.mask_not_borders = np.logical_not(self.mask_borders)

        self.errors_borders = []
        self.errors_not_borders = []

    def _run(self):

        true_Q_table = self.true.Q_table
        true_V_table = self.true.V_table
        true_policy = self.true.policy
        if true_Q_table is None:
            return

        Q_table = self.q.Q_table
        V_table = self.q.V_table
        policy = self.q.policy
        if Q_table is None:
            return

        T, C, nA = self.env.T, self.env.C, self.env.action_space.n

        self.replays.append(self.agent.replay_count)
        self.errors_Q_table.append(self._mse(true_Q_table, Q_table))
        self.errors_V_table.append(self._mse(true_V_table, V_table))
        self.errors_policy.append(self._mse(true_policy.reshape(T, C), policy.reshape(T, C)))

        self.errors_borders.append(self._mse(true_Q_table[self.mask_borders], Q_table[self.mask_borders]))
        self.errors_not_borders.append(self._mse(true_Q_table[self.mask_not_borders], Q_table[self.mask_not_borders]))

        print("Difference with the true Q-table")
        print(abs(true_Q_table.reshape(T, C, nA) - Q_table.reshape(T, C, nA)))

        print("Difference with the true V-table")
        print(abs(true_V_table.reshape(T, C) - V_table.reshape(T, C)))

        print("Difference with the true Policy")
        print(abs(true_policy.reshape(T, C) - policy.reshape(T, C)))

    def reset(self, agent):
        self.agent = agent

        self.replays = []
        self.errors_Q_table = []
        self.errors_V_table = []
        self.errors_policy = []

    def _mse(self, A, B):
        return np.sqrt(np.square(A.flatten() - B.flatten()).sum())


class QErrorDisplay(Callback):

    def __init__(self, condition, agent, q_error):
        super().__init__(condition, agent)
        self.q_error = q_error

    def _run(self):
        plt.figure()
        plt.plot(self.q_error.replays, self.q_error.errors_Q_table, '-o')
        plt.xlabel("Epochs")
        plt.ylabel("Difference with the true Q-table")
        plt.show()

        plt.figure()
        plt.plot(self.q_error.replays, self.q_error.errors_V_table, '-o')
        plt.xlabel("Epochs")
        plt.ylabel("Difference with the true V-table")
        plt.show()

        plt.plot(self.q_error.replays, self.q_error.errors_policy, '-o')
        plt.xlabel("Epochs")
        plt.ylabel("Difference with the policy")
        plt.show()

        plt.figure()
        plt.plot(self.q_error.replays, self.q_error.errors_borders, self.q_error.replays,
                 self.q_error.errors_not_borders)
        plt.legend(["Error at the borders", "Error not at the borders"])
        plt.xlabel("Epochs")
        plt.ylabel("Difference of each coefficient of the Q-table with the true Q-table")
        plt.show()


class RevenueMonitor(Callback):

    def __init__(self, condition, agent, q_compute, N, name="revenue_compute"):
        super().__init__(condition, agent)
        self.replays = []
        self.revenues = []
        self.q = q_compute
        self.N = N

        self.name = name

    def _run(self):
        policy = self.q.policy
        if policy is None:
            return

        if self.name == "true_revenue":
            revenue = average_n_episodes(self.env, policy.flatten(), self.N, self.agent.epsilon_min)
        else:
            revenue = average_n_episodes(self.env, policy.flatten(), self.N)
        self.replays.append(self.agent.replay_count)
        self.revenues.append(revenue)

        print("Average reward over {} episodes after {} replay : {}".format(self.N, self.agent.replay_count, revenue))

    def reset(self, agent):
        self.agent = agent

        self.replays = []
        self.revenues = []


class RevenueDisplay(Callback):

    def __init__(self, condition, agent, revenue_compute, reference=None):
        super().__init__(condition, agent)
        self.revenue = revenue_compute
        self.reference = reference

    def _run(self):
        plt.figure()
        plt.plot(self.revenue.replays, self.revenue.revenues, '-o')
        plt.xlabel("Epochs")
        plt.ylabel("Average revenue")
        plt.title("Average revenue over {} episodes".format(self.revenue.N))
        if self.reference is not None:
            X = self.revenue.replays[0], self.revenue.replays[-1]
            Y = [self.reference.revenues[0]] * 2
            plt.plot(X, Y, c='red', lw=3)
        plt.show()


class VDisplay(Callback):

    def __init__(self, condition, agent, q_compute):
        super().__init__(condition, agent)
        self.q = q_compute

    def _run(self):
        V_table = self.q.V_table
        if V_table is not None:
            visualisation_value_RM(V_table, self.env.T, self.env.C)


class PolicyDisplay(Callback):

    def __init__(self, condition, agent, q_compute):
        super().__init__(condition, agent)
        self.q = q_compute

    def _run(self):
        policy = self.q.policy
        if policy is not None:
            visualize_policy_RM(policy, self.env.T, self.env.C)


class MemoryMonitor(Callback):

    def __init__(self, condition, agent):
        super().__init__(condition, agent)
        self.replays = []
        self.hist2d = []

        self.name = "memory_monitor"

    def _run(self):
        self.replays.append(self.agent.replay_count)
        states = [self.env.nS - 1]
        actions = [0]
        weights = [0]
        for k in range(len(self.agent.memory)):
            states.append(self.env.to_idx(self.agent.memory[k][0][0][0], self.agent.memory[k][0][0][1]))
            actions.append(self.agent.memory[k][1])
            weights.append(1)
        h, xedges, yedges = np.histogram2d(states, actions, bins=[max(states) + 1, self.env.action_space.n],
                                           weights=weights)
        self.hist2d.append((h, xedges, yedges))

    def reset(self, agent):
        self.agent = agent

        self.replays = []
        self.hist2d=[]


class MemoryDisplay(Callback):

    def __init__(self, condition, agent, memory_monitor):
        super().__init__(condition, agent)
        self.memory_monitor = memory_monitor

    def _run(self):
        fig, ax = plt.subplots(tight_layout=True)
        h, xedges, yedges = self.memory_monitor.hist2d[-1]
        im = plt.pcolormesh(xedges, yedges, h.T)
        ax.set_xlim(xedges[0], xedges[-1])
        ax.set_ylim(yedges[0], yedges[-1])
        plt.xlabel("State index")
        plt.ylabel("Action index")
        plt.title("Transitions present in the agent's memory")
        plt.colorbar(im, ax=ax)
        plt.show()


class BatchMonitor(Callback):

    def __init__(self, condition, agent):
        super().__init__(condition, agent)
        self.replays = []
        self.hist2d = []

        self.name = "batch_monitor"

    def _run(self):
        self.replays.append(self.agent.replay_count)
        states = [self.env.nS - 1]
        actions = [0]
        weights = [0]
        L = list(self.agent.last_visited)
        for k in range(len(L)):
            states.append(self.env.to_idx(L[k][0][0], L[k][0][1]))
            actions.append(L[k][1])
            weights.append(1)
        h, xedges, yedges = np.histogram2d(states, actions, bins=[max(states) + 1, self.env.action_space.n],
                                           weights=weights)
        self.hist2d.append((h, xedges, yedges))


class BatchDisplay(Callback):

    def __init__(self, condition, agent, batch_monitor):
        super().__init__(condition, agent)
        self.batch_monitor = batch_monitor

    def _run(self):
        fig, ax = plt.subplots(tight_layout=True)
        h, xedges, yedges = self.batch_monitor.hist2d[-1]
        im = plt.pcolormesh(xedges, yedges, h.T)
        ax.set_xlim(xedges[0], xedges[-1])
        ax.set_ylim(yedges[0], yedges[-1])
        plt.xlabel("State index")
        plt.ylabel("Action index")
        plt.title("Transitions present in the agent's minibatch")
        plt.colorbar(im, ax=ax)
        plt.show()


class TotalBatchDisplay(Callback):

    def __init__(self, condition, agent, batch_monitor):
        super().__init__(condition, agent)
        self.batch_monitor = batch_monitor

    def _run(self):
        fig, ax = plt.subplots(tight_layout=True)
        H = np.zeros((self.batch_monitor.hist2d[0][0].shape[0], self.batch_monitor.hist2d[0][0].shape[1]), float)
        h, xedges, yedges = self.batch_monitor.hist2d[-1]
        for k in range(len(self.batch_monitor.hist2d)):
            H += self.batch_monitor.hist2d[k][0]
        im = plt.pcolormesh(xedges, yedges, H.T)
        ax.set_xlim(xedges[0], xedges[-1])
        ax.set_ylim(yedges[0], yedges[-1])
        plt.xlabel("State index")
        plt.ylabel("Action index")
        plt.title("Transitions picked up by the agent during experience replay")
        plt.colorbar(im, ax=ax)
        plt.show()


class SumtreeMonitor(Callback):

    def __init__(self, condition, agent):
        super().__init__(condition, agent)
        self.replays = []
        self.hist2d = []

        self.name = "sumtree_monitor"

    def _run(self):
        self.replays.append(self.agent.replay_count)
        states = [self.env.nS - 1]
        actions = [0]
        weights = [0]
        L = self.agent.tree.data
        for k in range(len(L)):
            if L[k] == 0:
                break
            states.append(self.env.to_idx(L[k][0][0][0], L[k][0][0][1]))
            actions.append(L[k][1])
            weights.append(self.agent.tree.tree[k + self.agent.tree.capacity - 1])
        h, xedges, yedges = np.histogram2d(states, actions, bins=[max(states) + 1, self.env.action_space.n],
                                           weights=weights)
        self.hist2d.append((h, xedges, yedges))


class SumtreeDisplay(Callback):

    def __init__(self, condition, agent, sumtree_monitor):
        super().__init__(condition, agent)
        self.sumtree_monitor = sumtree_monitor

    def _run(self):
        fig, ax = plt.subplots(tight_layout=True)
        h, xedges, yedges = self.sumtree_monitor.hist2d[-1]
        im = plt.pcolormesh(xedges, yedges, h.T)
        ax.set_xlim(xedges[0], xedges[-1])
        ax.set_ylim(yedges[0], yedges[-1])
        plt.xlabel("State index")
        plt.ylabel("Action index")
        plt.title("Priorities of the transitions stored in the agent's sumtree")
        plt.colorbar(im, ax=ax)
        plt.show()
