import torch, random
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.distributions import Bernoulli
from typing import List, Dict, Any

# === Base Class ===
class Player:
    def reset(self): pass
    def move(self, my_id: str, history: List[Dict[str, Any]]) -> int: pass
    def update_rewards(self, r): pass
    def update_policy(self): pass

# === Strategy Classes ===
class TitForTat(Player):
    def move(self, my_id, history):
        if not history: return 1
        opp_id = [pid for pid in history[-1] if pid not in ['round', my_id]][0]
        return history[-1][opp_id]

class GenerousTitForTat(Player):
    def move(self, my_id, history):
        if not history: return 1
        opp_id = [pid for pid in history[-1] if pid not in ['round', my_id]][0]
        return history[-1][opp_id] if random.random() < 0.9 else 1

class TitForTwoTats(Player):
    def move(self, my_id, history):
        if len(history) < 2: return 1
        last_two = [h[next(pid for pid in h if pid != 'round' and pid != my_id)] for h in history[-2:]]
        return 0 if last_two == [0, 0] else 1

class GrimTrigger(Player):
    def reset(self): self.triggered = False
    def move(self, my_id, history):
        if self.triggered: return 0
        for h in history:
            opp_id = [pid for pid in h if pid not in ['round', my_id]][0]
            if h[opp_id] == 0:
                self.triggered = True
                return 0
        return 1
class Cooperator(Player):
    def move(self, my_id, history):
        return 1
    
class AlwaysDefect(Player):
    def move(self, my_id, history):
        return 0

class RandomStrategy(Player):
    def move(self, my_id, history):
        return random.choice([0, 1])

class Pavlov(Player):
    def reset(self): pass
    def move(self, my_id, history):
        if not history: return 1
        last = history[-1]
        opp_id = [pid for pid in last if pid not in ['round', my_id]][0]
        m, o = last[my_id], last[opp_id]
        return 1 if m == o else 0
class Detective(Player):
    def reset(self):
        self.sequence = [1, 0, 1, 1]  # C, D, C, C
        self.turn = 0
        self.use_tft = False
        self.opponent_moves = []

    def move(self, my_id, history):
        if self.turn < 4:
            action = self.sequence[self.turn]
        else:
            if not self.use_tft:
                if 0 in self.opponent_moves:
                    self.use_tft = True
                else:
                    return 0  # AlwaysDefect if opponent never defected
            # Switch to TitForTat behavior
            action = 1 if self.opponent_moves[-1] == 1 else 0
        if history:
            opp_id = next(k for k in history[-1] if k not in ['round', my_id])
            self.opponent_moves.append(history[-1][opp_id])
        self.turn += 1
        return action

# === FFN Agent with 5-round input and entropy regularization ===
class FFNAgent(Player):
    def __init__(self, name = "FFNAgent", lr=0.01):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(10, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid())
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.log_probs, self.rewards = [], []
        self.name = name

    def reset(self):
        self.log_probs, self.rewards = [], []

    def move(self, my_id, history):
        state = self.encode_state(my_id, history)
        prob = self.net(state)
        m = Bernoulli(prob)
        action = m.sample()
        self.log_probs.append((m.log_prob(action), m.entropy()))
        return int(action.item())

    def encode_state(self, my_id, history):
        state = []
        opp_id = [pid for pid in history[-1].keys() if pid not in ['round', my_id]][0] if history else None
        for i in range(1, 6):
            if len(history) >= i:
                h = history[-i]
                state.extend([h.get(my_id, 0), h.get(opp_id, 0)])
            else:
                state.extend([0, 0])
        return torch.tensor(state, dtype=torch.float32)

    def update_rewards(self, r):
        self.rewards.append(r)

    def update_policy(self):
        if not self.log_probs: return
        R, returns = 0, []
        for r in reversed(self.rewards):
            R = r + 0.99 * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)
        baseline = returns.mean()
        adv = returns - baseline
        loss = -torch.stack([lp * a - 0.01 * ent for (lp, ent), a in zip(self.log_probs, adv)]).sum()
        self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
        self.reset()

# === RNN Agent with reward shaping and entropy ===
class RNNAgent(Player):
    def __init__(self, name = "RNNAgent" , input_size=2, hidden_size=16, lr=0.005):
        super().__init__()
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        self.output = nn.Sequential(nn.Linear(hidden_size, 1), nn.Sigmoid())
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.rewards, self.log_probs = [], []
        self.h = None
        self.prev_action = 1
        self.name = name

    def parameters(self):
        return list(self.rnn.parameters()) + list(self.output.parameters())

    def reset(self):
        self.h = None
        self.rewards.clear()
        self.log_probs.clear()
        self.prev_action = 1

    def move(self, my_id, history):
        x_seq = self.build_input_sequence(my_id, history)
        out, self.h = self.rnn(x_seq, self.h)
        prob = self.output(out[:, -1, :])
        m = Bernoulli(prob)
        action = m.sample()
        self.log_probs.append((m.log_prob(action), m.entropy()))
        self.prev_action = int(action.item())
        return self.prev_action

    def build_input_sequence(self, my_id, history):
        inputs = []
        opp_id = [pid for pid in history[-1] if pid not in ['round', my_id]][0] if history else 'none'
        for h in history[-10:]:
            inputs.append([h.get(my_id, 0), h.get(opp_id, 0)])
        while len(inputs) < 10:
            inputs.insert(0, [0, 0])
        return torch.tensor([inputs], dtype=torch.float32)

    def update_rewards(self, r): self.rewards.append(r)

    def update_policy(self):
        if not self.log_probs: return
        G, returns = 0, []
        for r in reversed(self.rewards):
            G = r + 0.99 * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)
        baseline = returns.mean()
        adv = returns - baseline
        loss = -torch.stack([lp * a - 0.01 * ent for (lp, ent), a in zip(self.log_probs, adv)]).sum()
        self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
        self.reset()

# === Match Simulation ===
def run_match(players: Dict[str, Player], rounds=200, training=True):
    for p in players.values(): p.reset()
    history = []; rewards = {pid: 0 for pid in players}
    pids = list(players.keys())
    for r in range(1, rounds+1):
        round_moves = {'round': r}
        for pid in pids:
            round_moves[pid] = players[pid].move(pid, history)
        history.append(round_moves)
        a1, a2 = round_moves[pids[0]], round_moves[pids[1]]
        r1, r2 = (3, 3) if (a1, a2) == (1, 1) else (0, 5) if a1 == 1 else (5, 0) if a2 == 1 else (1, 1)
        rewards[pids[0]] += r1
        rewards[pids[1]] += r2
        if training:
            for pid, r in zip(pids, [r1, r2]):
                players[pid].update_rewards(r)
    return history, rewards

# === Curriculum Trainer ===
def train_with_curriculum(agent: Player, curriculum_fn, total_eps=20000):
    for ep in range(total_eps):
        opp_class = curriculum_fn(ep)
        opp = opp_class()
        run_match({'A': agent, 'B': opp}, training=True)
        agent.update_policy()
        if ((ep + 1) % 2000 == 0) or (ep == 0):
            print(f"Episode {ep+1}/{total_eps} for {agent.name}")

# === Curriculum Strategy ===
def curriculum(ep):
    if ep < 3000:
        return random.choice([TitForTat, Cooperator, Detective])
    elif ep < 6000:
        return random.choice([Pavlov, TitForTwoTats, RandomStrategy])
    elif ep < 9000:
        return random.choice([GrimTrigger, AlwaysDefect])
    else:
        return random.choice([TitForTat, Detective, GrimTrigger, Cooperator, AlwaysDefect, RandomStrategy])


# === Run it ===
if __name__ == '__main__':
    agent_rnn = RNNAgent()
    agent_ffn = FFNAgent()
    for ep in range(20000):
        strat = curriculum(ep)()
        run_match({'A': agent_rnn, 'B': strat}, training=True)
        agent_rnn.update_policy()

        strat = curriculum(ep)()
        run_match({'A': agent_ffn, 'B': strat}, training=True)
        agent_ffn.update_policy()

        if ((ep+1) % 1000 == 0) or (ep == 0):
            print(f"Episode {ep+1}/20000: RNN and FFN training vs curriculum strategies")

    for i, agent in enumerate([agent_rnn, agent_ffn], start=1):
        for strat in [GenerousTitForTat, Pavlov, TitForTat, TitForTwoTats, GrimTrigger, RandomStrategy]:
            print(f"Evaluating {agent.name} vs {strat.__name__}")
            h, rewards = run_match({'A': agent, 'B' : strat()}, training=False)
            plt.figure(figsize=(8, 3))
            plt.step(range(len(h)), [x['A'] for x in h], where='post', label=f'Agent {i}')
            plt.step(range(len(h)), [x['B'] for x in h], where='post', label=strat.__name__, linestyle='--')
            plt.title(f"Agent {i} vs {strat.__name__}: Agent {rewards['A']:.1f} | Opp {rewards['B']:.1f}")
            plt.yticks([0,1], ['Defect','Cooperate'])
            plt.xlabel("Round"); plt.legend(); plt.grid(True); plt.tight_layout()
            plt.show()

