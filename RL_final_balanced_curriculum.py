import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random
from torch.distributions import Bernoulli
from typing import List, Dict, Any

# === Base Class ===
class Player:
    def reset(self):
        pass
    def move(self, my_id: str, history: List[Dict[str, Any]]) -> int:
        pass

# === FFN Agent ===
class FFNAgent(Player):
    def __init__(self, lr=0.01):
        super().__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(6, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.log_probs, self.rewards = [], []

    def reset(self):
        self.log_probs, self.rewards = [], []

    def move(self, my_id, history):
        state = self.encode_state(my_id, history)
        prob = self.policy_net(state)
        m = Bernoulli(prob)
        action = m.sample()
        self.log_probs.append(m.log_prob(action))
        return int(action.item())

    def encode_state(self, my_id, history):
        state = []
        opp_id = [pid for pid in history[-1].keys() if pid not in ['round', my_id]][0] if history else 'none'
        for i in range(1, 4):
            if len(history) >= i:
                h = history[-i]
                state.extend([h.get(my_id, 0), h.get(opp_id, 0)])
            else:
                state.extend([0, 0])
        return torch.tensor(state, dtype=torch.float32)

    def update_policy(self):
        if not self.log_probs or not self.rewards:
            return
        R, returns = 0, []
        for r in reversed(self.rewards):
            R = r + 0.99 * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)
        baseline = returns.mean()
        advantages = returns - baseline
        loss_terms = [-log_prob * adv for log_prob, adv in zip(self.log_probs, advantages)]
        if not loss_terms:
            return
        loss = torch.stack(loss_terms).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.log_probs, self.rewards = [], []

    def update_rewards(self, r):
        self.rewards.append(r)

# === RNN Agent ===
class RNNAgent(Player):
    def __init__(self, lr=0.01):
        super().__init__()
        self.rnn = nn.RNN(input_size=2, hidden_size=16, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        self.hidden = None
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.log_probs, self.rewards = [], []

    def parameters(self):
        return list(self.rnn.parameters()) + list(self.fc.parameters())

    def reset(self):
        self.hidden = None
        self.log_probs, self.rewards = [], []

    def move(self, my_id, history):
        state_seq = self.encode_sequence(my_id, history)
        out, self.hidden = self.rnn(state_seq, self.hidden)
        prob = self.fc(out[:, -1, :])
        m = Bernoulli(prob)
        action = m.sample()
        self.log_probs.append(m.log_prob(action.squeeze()))
        return int(action.item())

    def encode_sequence(self, my_id, history):
        seq = []
        opp_id = [pid for pid in history[-1].keys() if pid not in ['round', my_id]][0] if history else 'none'
        for h in history:
            seq.append([h.get(my_id, 0), h.get(opp_id, 0)])
        if not seq:
            seq.append([0, 0])
        return torch.tensor([seq], dtype=torch.float32)

    def update_policy(self):
        if not self.log_probs or not self.rewards:
            return
        R, returns = 0, []
        for r in reversed(self.rewards):
            R = r + 0.99 * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)
        baseline = returns.mean()
        advantages = returns - baseline
        loss_terms = [-log_prob * adv for log_prob, adv in zip(self.log_probs, advantages)]
        if not loss_terms:
            return
        loss = torch.stack(loss_terms).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.log_probs, self.rewards = [], []

    def update_rewards(self, r):
        self.rewards.append(r)

# === Fixed Strategies ===
class TitForTat(Player):
    def move(self, my_id, history):
        if not history:
            return 1
        opp_id = [pid for pid in history[-1] if pid not in ['round', my_id]][0]
        return history[-1][opp_id]

class AlwaysCooperate(Player):
    def move(self, my_id, history):
        return 1

class AlwaysDefect(Player):
    def move(self, my_id, history):
        return 0

class GrimTrigger(Player):
    def reset(self):
        self.triggered = False
    def move(self, my_id, history):
        if self.triggered:
            return 0
        for h in history:
            opp_id = [pid for pid in h if pid not in ['round', my_id]][0]
            if h[opp_id] == 0:
                self.triggered = True
                return 0
        return 1

class SuspiciousTitForTat(Player):
    def move(self, my_id, history):
        if not history:
            return 0
        opp_id = [pid for pid in history[-1] if pid not in ['round', my_id]][0]
        return history[-1][opp_id]

class RandomStrategy(Player):
    def move(self, my_id, history):
        return random.choice([0, 1])

class Pavlov(Player):
    def reset(self):
        self.last_move = 1
    def move(self, my_id, history):
        if not history:
            return 1
        opp_id = [pid for pid in history[-1] if pid not in ['round', my_id]][0]
        last_self = history[-1][my_id]
        last_opp = history[-1][opp_id]
        self.last_move = 1 if last_self == last_opp else 0
        return self.last_move

# === Match Simulation ===
def run_match(players: Dict[str, Player], rounds: int, training: bool, update=True):
    for p in players.values():
        p.reset()
    history = []
    rewards = {pid: 0 for pid in players.keys()}
    pids = list(players.keys())
    for r in range(1, rounds + 1):
        round_moves = {'round': r}
        for pid in pids:
            action = players[pid].move(pid, history)
            round_moves[pid] = action
        history.append(round_moves)
        a1, a2 = round_moves[pids[0]], round_moves[pids[1]]
        r1, r2 = (3, 3) if (a1, a2) == (1, 1) else (0, 5) if a1 == 1 else (5, 0) if a2 == 1 else (1, 1)
        rewards[pids[0]] += r1
        rewards[pids[1]] += r2
        if training and update:
            for pid, r in zip(pids, [r1, r2]):
                if hasattr(players[pid], 'update_rewards'):
                    players[pid].update_rewards(r)
    return history, rewards

# === Curriculum Training ===
def curriculum_train(ffn, rnn, curriculum, phase_episodes=1500, rounds=150):
    for phase_num, phase_pool in enumerate(curriculum, 1):
        print(f"\n=== Phase {phase_num}: {', '.join(type(s).__name__ for s in phase_pool)} ===")
        for ep in range(phase_episodes):
            opp = random.choice(phase_pool)
            run_match({'FFN': ffn, 'S': opp}, rounds, training=True)
            run_match({'RNN': rnn, 'S': opp}, rounds, training=True)
            ffn.update_policy()
            rnn.update_policy()
            if (ep + 1) % 500 == 0:
                print(f"Episode {ep + 1}/{phase_episodes}")

# === Evaluation ===
def evaluate_and_plot_individual(agent_ffn, agent_rnn, strategies, rounds=200):
    def plot_match(agent_name, strategy_name, moves_agent, moves_strategy, avg_a, avg_s):
        plt.figure(figsize=(10, 5))
        plt.step(range(1, rounds + 1), moves_agent, label=agent_name)
        plt.step(range(1, rounds + 1), moves_strategy, label=strategy_name, linestyle='dotted')
        plt.title(f"{agent_name} vs {strategy_name}\nAvg Scores: {agent_name}={avg_a:.2f}, {strategy_name}={avg_s:.2f}")
        plt.ylim(-0.1, 1.1)
        plt.xlabel("Round")
        plt.ylabel("Action (1=Cooperate, 0=Defect)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{agent_name}_vs_{strategy_name}.png")
        plt.close()

    for strat in strategies:
        strat_name = type(strat).__name__
        history_f, rewards_f = run_match({'FFN': agent_ffn, 'S': strat}, rounds, training=False)
        moves_ffn = [h['FFN'] for h in history_f]
        moves_s1 = [h['S'] for h in history_f]
        plot_match("FFN", strat_name, moves_ffn, moves_s1, rewards_f['FFN'] / rounds, rewards_f['S'] / rounds)

        history_r, rewards_r = run_match({'RNN': agent_rnn, 'S': strat}, rounds, training=False)
        moves_rnn = [h['RNN'] for h in history_r]
        moves_s2 = [h['S'] for h in history_r]
        plot_match("RNN", strat_name, moves_rnn, moves_s2, rewards_r['RNN'] / rounds, rewards_r['S'] / rounds)

    history_both, rewards_both = run_match({'FFN': agent_ffn, 'RNN': agent_rnn}, rounds, training=False)
    moves_ffn = [h['FFN'] for h in history_both]
    moves_rnn = [h['RNN'] for h in history_both]
    plot_match("FFN", "RNN", moves_ffn, moves_rnn, rewards_both['FFN'] / rounds, rewards_both['RNN'] / rounds)
    plot_match("RNN", "FFN", moves_rnn, moves_ffn, rewards_both['RNN'] / rounds, rewards_both['FFN'] / rounds)

# === Main Execution ===
if __name__ == '__main__':
    ffn = FFNAgent()
    rnn = RNNAgent()

    curriculum = [
        [TitForTat(), Pavlov()],
        [GrimTrigger(), SuspiciousTitForTat()],
        [Pavlov(), SuspiciousTitForTat(), RandomStrategy()],
        [TitForTat(), Pavlov(), GrimTrigger(), SuspiciousTitForTat(), RandomStrategy(), AlwaysDefect()]
    ]

    print("Training agents with balanced curriculum...")
    curriculum_train(ffn, rnn, curriculum)

    strategies = [TitForTat(), GrimTrigger(), Pavlov(), AlwaysCooperate(), AlwaysDefect(), SuspiciousTitForTat(), RandomStrategy()]
    print("Final evaluation...")
    evaluate_and_plot_individual(ffn, rnn, strategies)

    torch.save(ffn.policy_net.state_dict(), "ffn_model.pth")
    torch.save(rnn.rnn.state_dict(), "rnn_rnn.pth")
    torch.save(rnn.fc.state_dict(), "rnn_fc.pth")
    print("Models saved.")
