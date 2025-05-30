# pip install torch numpy matplotlib scipy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Bernoulli
import numpy as np
import matplotlib.pyplot as plt
import random
from typing import Dict, List, Union, Any
from scipy.stats import entropy

class Player:
    """Base class for all players in the iterated game."""
    def reset(self):
        """Reset player state before a new match."""
        pass
    
    def move(self, my_id: str, history: List[Dict[str, Any]]) -> int:
        """
        Decide on next move based on game history.
        
        Args:
            my_id: This player's unique identifier in the game
            history: List of round dictionaries containing all players' previous moves
            
        Returns:
            int: Action (1=Cooperate, 0=Defect)
        """
        pass


class MLPAgent(Player):
    """
    Reinforcement Learning Agent with MLP policy network.
    Uses last 4 moves of each player as state.
    Uses REINFORCE algorithm with entropy regularization for policy updates.
    """
    def __init__(self, lr=0.01, entropy_weight=0.01):
        """
        Initialize agent with policy network and optimizer.
        
        Args:
            lr: Learning rate for Adam optimizer
            entropy_weight: Weight for entropy regularization
        """
        super().__init__()
        # Policy network: 8 inputs (last 4 moves of self and opponent) -> 16 hidden -> 1 output
        self.policy_net = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.log_probs = []
        self.entropies = []
        self.rewards = []
        self.last_action = None
        self.entropy_weight = entropy_weight

    def reset(self):
        """Reset agent's memory before a new match."""
        self.log_probs = []
        self.entropies = []
        self.rewards = []
        self.last_action = None

    def move(self, my_id: str, history: List[Dict[str, Any]]) -> int:
        """
        Choose action based on policy network.
        
        Args:
            my_id: This agent's unique identifier in the game
            history: List of round dictionaries containing all players' previous moves
            
        Returns:
            int: Action (1=Cooperate, 0=Defect)
        """
        # Get last 4 moves of self and opponent
        my_moves = []
        opp_moves = []
        
        if not history:
            # If no history, default to all cooperation
            state = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.float32)
        else:
            # Find opponent ID
            opp_id = [pid for pid in history[0].keys() if pid not in ['round', my_id]][0]
            
            # Get last 4 moves (or pad with 1s if fewer than 4 rounds played)
            for i in range(min(4, len(history))):
                idx = len(history) - 1 - i
                if idx >= 0:
                    my_moves.insert(0, history[idx].get(my_id, 1))
                    opp_moves.insert(0, history[idx].get(opp_id, 1))
            
            # Pad with 1s if needed
            while len(my_moves) < 4:
                my_moves.insert(0, 1)
            while len(opp_moves) < 4:
                opp_moves.insert(0, 1)
            
            # Create state tensor
            state = torch.tensor(my_moves + opp_moves, dtype=torch.float32)
        
        # Forward pass through policy network
        prob = self.policy_net(state)
        m = Bernoulli(prob)
        action = m.sample()
        
        # Calculate entropy of the policy
        entropy_val = -(prob * torch.log(prob + 1e-10) + (1 - prob) * torch.log(1 - prob + 1e-10))
        
        # Store log probability and entropy for REINFORCE update
        self.log_probs.append(m.log_prob(action))
        self.entropies.append(entropy_val)
        self.last_action = int(action.item())
        return self.last_action

    def update_reward(self, reward):
        """
        Update the agent's rewards.
        
        Args:
            reward: The reward received in the current round
        """
        self.rewards.append(reward)

    def update_policy(self):
        """
        Update policy using REINFORCE algorithm with entropy regularization.
        """
        if not self.rewards:
            return  # No rewards to learn from
            
        # Calculate returns with discount factor 0.99
        R = 0
        returns = []
        for r in reversed(self.rewards):
            R = r + 0.99 * R
            returns.insert(0, R)
            
        # Normalize returns for stable learning
        returns = torch.tensor(returns)
        if len(returns) > 1:  # Only normalize if we have multiple returns
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        # Calculate loss with entropy regularization
        policy_loss = 0
        entropy_loss = 0
        for log_prob, R, entropy_val in zip(self.log_probs, returns, self.entropies):
            policy_loss -= log_prob * R
            entropy_loss -= entropy_val  # Maximize entropy (encourage exploration)
        
        # Combine losses
        total_loss = policy_loss - self.entropy_weight * entropy_loss

        # Gradient descent step
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Clear memory
        self.log_probs = []
        self.entropies = []
        self.rewards = []


class RNNAgent(Player):
    """
    Reinforcement Learning Agent with RNN policy network.
    Uses REINFORCE algorithm with entropy regularization for policy updates.
    """
    def __init__(self, input_size=2, hidden_size=16, lr=0.01, entropy_weight=0.01):
        """
        Initialize agent with RNN policy network and optimizer.
        
        Args:
            input_size: Size of input vector (self move, opponent move)
            hidden_size: Size of hidden state
            lr: Learning rate for Adam optimizer
            entropy_weight: Weight for entropy regularization
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.entropy_weight = entropy_weight
        
        # RNN cell
        self.rnn = nn.GRUCell(input_size, hidden_size)
        
        # Output layer
        self.output = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        self.optimizer = optim.Adam(list(self.rnn.parameters()) + list(self.output.parameters()), lr=lr)
        self.log_probs = []
        self.entropies = []
        self.rewards = []
        self.last_action = None
        self.hidden_state = None

    def reset(self):
        """Reset agent's memory before a new match."""
        self.log_probs = []
        self.entropies = []
        self.rewards = []
        self.last_action = None
        self.hidden_state = torch.zeros(1, self.hidden_size)

    def move(self, my_id: str, history: List[Dict[str, Any]]) -> int:
        """
        Choose action based on RNN policy network.
        
        Args:
            my_id: This agent's unique identifier in the game
            history: List of round dictionaries containing all players' previous moves
            
        Returns:
            int: Action (1=Cooperate, 0=Defect)
        """
        # Initialize hidden state if None
        if self.hidden_state is None:
            self.hidden_state = torch.zeros(1, self.hidden_size)
        
        # Get previous moves
        if not history:
            # If no history, default to cooperation
            my_prev = 1
            opp_prev = 1
        else:
            # Find opponent ID
            opp_id = [pid for pid in history[-1].keys() if pid not in ['round', my_id]][0]
            
            # Get last moves
            my_prev = history[-1].get(my_id, 1)
            opp_prev = history[-1].get(opp_id, 1)
        
        # Create input tensor
        x = torch.tensor([[my_prev, opp_prev]], dtype=torch.float32)
        
        # Update hidden state
        self.hidden_state = self.rnn(x, self.hidden_state)
        
        # Get action probability
        prob = self.output(self.hidden_state)
        m = Bernoulli(prob)
        action = m.sample()
        
        # Calculate entropy of the policy
        entropy_val = -(prob * torch.log(prob + 1e-10) + (1 - prob) * torch.log(1 - prob + 1e-10))
        
        # Store log probability and entropy for REINFORCE update
        self.log_probs.append(m.log_prob(action))
        self.entropies.append(entropy_val)
        self.last_action = int(action.item())
        return self.last_action

    def update_reward(self, reward):
        """
        Update the agent's rewards.
        
        Args:
            reward: The reward received in the current round
        """
        self.rewards.append(reward)

    def update_policy(self):
        """
        Update policy using REINFORCE algorithm with entropy regularization.
        """
        if not self.rewards:
            return  # No rewards to learn from
            
        # Calculate returns with discount factor 0.99
        R = 0
        returns = []
        for r in reversed(self.rewards):
            R = r + 0.99 * R
            returns.insert(0, R)
            
        # Normalize returns for stable learning
        returns = torch.tensor(returns)
        if len(returns) > 1:  # Only normalize if we have multiple returns
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        # Calculate loss with entropy regularization
        policy_loss = 0
        entropy_loss = 0
        for log_prob, R, entropy_val in zip(self.log_probs, returns, self.entropies):
            policy_loss -= log_prob * R
            entropy_loss -= entropy_val  # Maximize entropy (encourage exploration)
        
        # Combine losses
        total_loss = policy_loss - self.entropy_weight * entropy_loss

        # Gradient descent step
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Clear memory
        self.log_probs = []
        self.entropies = []
        self.rewards = []


class AlwaysDefect(Player):
    """Always defects."""
    def move(self, my_id: str, history: List[Dict[str, Any]]) -> int:
        return 0


class AlwaysCooperate(Player):
    """Always cooperates."""
    def move(self, my_id: str, history: List[Dict[str, Any]]) -> int:
        return 1


class GrimTrigger(Player):
    """Cooperates until opponent defects, then always defects."""
    def reset(self):
        self.triggered = False
        
    def move(self, my_id: str, history: List[Dict[str, Any]]) -> int:
        if not history:
            return 1  # Cooperate on first move
            
        if self.triggered:
            return 0  # Always defect once triggered
            
        # Find opponent's ID
        opp_id = [pid for pid in history[-1].keys() if pid not in ['round', my_id]][0]
        
        # Check if opponent has ever defected
        for h in history:
            if opp_id in h and h[opp_id] == 0:
                self.triggered = True
                return 0
        return 1


class TitForTat(Player):
    """Cooperates on first move, then copies opponent's previous move."""
    def move(self, my_id: str, history: List[Dict[str, Any]]) -> int:
        if not history:
            return 1  # Cooperate on first move
        
        # Find opponent's ID and get their last move
        opp_id = [pid for pid in history[-1].keys() if pid not in ['round', my_id]][0]
        return history[-1][opp_id]


class SuspiciousTitForTat(Player):
    """Defects on first move, then copies opponent's previous move."""
    def move(self, my_id: str, history: List[Dict[str, Any]]) -> int:
        if not history:
            return 0  # Defect on first move
        
        # Find opponent's ID and get their last move
        opp_id = [pid for pid in history[-1].keys() if pid not in ['round', my_id]][0]
        return history[-1][opp_id]


class Pavlov(Player):
    """Win-Stay, Lose-Shift strategy."""
    def reset(self):
        self.last_move = 1
        
    def move(self, my_id: str, history: List[Dict[str, Any]]) -> int:
        if not history:
            self.last_move = 1
            return 1  # Cooperate on first move
            
        # Find opponent's ID
        opp_id = [pid for pid in history[-1].keys() if pid not in ['round', my_id]][0]
        
        last_self = history[-1][my_id]
        last_opp = history[-1][opp_id]
        
        # Win-Stay, Lose-Shift logic
        # Win: both cooperated or both defected
        # Lose: one cooperated and one defected
        if last_self == last_opp:  # Win (CC or DD)
            self.last_move = last_self  # Stay with same move
        else:  # Lose (CD or DC)
            self.last_move = 1 - last_self  # Shift to opposite move
            
        return self.last_move


class MajorityRule(Player):
    """Plays what the opponent has played most often."""
    def reset(self):
        self.opp_moves = []
        
    def move(self, my_id: str, history: List[Dict[str, Any]]) -> int:
        if not history:
            return 1  # Cooperate on first move
        
        # Find opponent's ID
        opp_id = [pid for pid in history[-1].keys() if pid not in ['round', my_id]][0]
        
        # Track opponent's moves
        self.opp_moves.append(history[-1][opp_id])
        
        # Play what opponent has played most often
        if sum(self.opp_moves) > len(self.opp_moves) / 2:
            return 1  # Cooperate if opponent mostly cooperates
        else:
            return 0  # Defect if opponent mostly defects


def run_match(players: Dict[str, Player], rounds: int, training: bool):
    """
    Run an iterated match among players.

    Args:
        players: dict mapping player IDs to Player instances
        rounds: number of rounds to play
        training: if True, agents can update their policies

    Returns:
        history: list of dicts recording moves each round
        rewards: dict of cumulative rewards per player
    """
    # Reset all players
    for p in players.values():
        p.reset()

    history = []
    rewards = {pid: 0 for pid in players.keys()}

    # For 2-player games, get the list of player IDs
    pids = list(players.keys())

    for r in range(1, rounds + 1):
        round_moves = {'round': r}

        # Each player chooses an action
        for pid in pids:
            action = players[pid].move(pid, history)
            round_moves[pid] = action

        history.append(round_moves)

        # Compute rewards for 2-player iterated game (Prisoner's Dilemma style)
        # Payoff matrix: Actions: 1=Cooperate, 0=Defect
        # Payoff matrix for player 1 (row) vs player 2 (col): 
        #    C    D
        # C  3,3  0,5
        # D  5,0  1,1
        if len(pids) == 2:
            pid1, pid2 = pids[0], pids[1]
            a1, a2 = round_moves[pid1], round_moves[pid2]
            
            # Payoffs
            if a1 == 1 and a2 == 1:  # Both cooperate
                r1, r2 = 3, 3
            elif a1 == 1 and a2 == 0:  # Player 1 cooperates, Player 2 defects
                r1, r2 = 0, 5
            elif a1 == 0 and a2 == 1:  # Player 1 defects, Player 2 cooperates
                r1, r2 = 5, 0
            else:  # Both defect
                r1, r2 = 1, 1
                
            rewards[pid1] += r1
            rewards[pid2] += r2
            
            # Update RL agents with rewards
            if training:
                if isinstance(players[pid1], (MLPAgent, RNNAgent)):
                    players[pid1].update_reward(r1)
                if isinstance(players[pid2], (MLPAgent, RNNAgent)):
                    players[pid2].update_reward(r2)
        else:
            # For >2 players, no payoff defined yet
            pass

    return history, rewards


def calculate_entropy(moves):
    """
    Calculate the entropy of a sequence of moves.
    
    Args:
        moves: List of moves (0s and 1s)
    
    Returns:
        float: Entropy value
    """
    if not moves:
        return 0
    
    # Count occurrences
    counts = [moves.count(0), moves.count(1)]
    
    # Calculate probabilities
    probs = [count / len(moves) for count in counts if count > 0]
    
    # Calculate entropy
    return entropy(probs, base=2)


def train_agents(agent1, agent2, strategies, episodes=10000, rounds=200):
    """
    Train two RL agents through both self-play and against fixed strategies.
    
    Args:
        agent1: First RL agent (MLP-based)
        agent2: Second RL agent (RNN-based)
        strategies: List of fixed strategy instances
        episodes: Number of episodes to train
        rounds: Number of rounds per episode
    """
    print(f"Training for {episodes} episodes...")
    
    # Track performance
    performance_history = {
        'self_play': [],
        'strategies': {type(strat).__name__: [] for strat in strategies}
    }
    
    for ep in range(episodes):
        if (ep + 1) % 1000 == 0:
            print(f"Episode {ep + 1}/{episodes}")
            
        # Every 5 episodes, play against a fixed strategy
        if ep % 5 == 0 and strategies:
            # Choose a random strategy to play against
            strategy = random.choice(strategies)
            strat_name = type(strategy).__name__
            
            # Train agent1 against the strategy
            history, rewards = run_match({'A1': agent1, 'S': strategy}, rounds, training=True)
            agent1.update_policy()
            
            # Track performance
            if (ep + 1) % 1000 == 0:
                performance_history['strategies'][strat_name].append(rewards['A1'])
                
            # Train agent2 against the strategy
            history, rewards = run_match({'A2': agent2, 'S': strategy}, rounds, training=True)
            agent2.update_policy()
        else:
            # Self-play training
            history, rewards = run_match({'A1': agent1, 'A2': agent2}, rounds, training=True)
            
            # Track performance in self-play
            if (ep + 1) % 1000 == 0:
                performance_history['self_play'].append((rewards['A1'], rewards['A2']))
            
            # Update policies
            agent1.update_policy()
            agent2.update_policy()
    
    return performance_history


def evaluate_and_plot(agent1, agent2, strategies, rounds=200):
    """
    Evaluate agents against fixed strategies and plot results.
    
    Args:
        agent1: First RL agent (MLP-based)
        agent2: Second RL agent (RNN-based)
        strategies: List of fixed strategy instances
        rounds: Number of rounds per evaluation match
    """
    # First, plot self-play match showing entire history
    history_sp, rewards_sp = run_match({'A1': agent1, 'A2': agent2}, rounds, training=False)
    moves_a1 = [h['A1'] for h in history_sp]
    moves_a2 = [h['A2'] for h in history_sp]

    # Calculate average scores and entropy
    avg_score_a1 = rewards_sp["A1"] / rounds
    avg_score_a2 = rewards_sp["A2"] / rounds
    entropy_a1 = calculate_entropy(moves_a1)
    entropy_a2 = calculate_entropy(moves_a2)

    plt.figure(figsize=(12, 6))
    plt.step(range(1, rounds + 1), moves_a1, label='Agent1 (MLP)')
    plt.step(range(1, rounds + 1), moves_a2, label='Agent2 (RNN)')
    plt.title(f'Self-play: Agent1 vs Agent2\nAvg Scores: A1={avg_score_a1:.2f}, A2={avg_score_a2:.2f}\nEntropy: A1={entropy_a1:.2f}, A2={entropy_a2:.2f}')
    plt.ylim(-0.1, 1.1)
    plt.xlabel('Round')
    plt.ylabel('Action (1=C, 0=D)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('agents_self_play.png')
    plt.show()

    # Create a figure for all strategy evaluations
    fig1, axs1 = plt.subplots(len(strategies), 1, figsize=(12, 4 * len(strategies)))
    fig2, axs2 = plt.subplots(len(strategies), 1, figsize=(12, 4 * len(strategies)))
    
    # Ensure axs is always a list even with a single strategy
    if len(strategies) == 1:
        axs1 = [axs1]
        axs2 = [axs2]

    # Evaluate against each strategy
    for i, strat in enumerate(strategies):
        strat_name = type(strat).__name__
        print(f"Evaluating against {strat_name}...")
        
        # Evaluate agent1 vs strategy
        history1, rewards1 = run_match({'A1': agent1, 'S': strat}, rounds, training=False)
        
        # Evaluate agent2 vs strategy
        history2, rewards2 = run_match({'A2': agent2, 'S': strat}, rounds, training=False)

        # Calculate average scores and entropy
        avg_score_a1 = rewards1["A1"] / rounds
        avg_score_s1 = rewards1["S"] / rounds
        avg_score_a2 = rewards2["A2"] / rounds
        avg_score_s2 = rewards2["S"] / rounds
        
        moves_a1 = [h['A1'] for h in history1]
        moves_s1 = [h['S'] for h in history1]
        moves_a2 = [h['A2'] for h in history2]
        moves_s2 = [h['S'] for h in history2]
        
        entropy_a1 = calculate_entropy(moves_a1)
        entropy_s1 = calculate_entropy(moves_s1)
        entropy_a2 = calculate_entropy(moves_a2)
        entropy_s2 = calculate_entropy(moves_s2)

        # Plot agent1 vs strategy
        axs1[i].step(range(1, rounds + 1), moves_a1, label='Agent1 (MLP)')
        axs1[i].step(range(1, rounds + 1), moves_s1, label=strat_name, linestyle='dotted')
        axs1[i].set_title(f'Agent1 vs {strat_name}\nAvg: A1={avg_score_a1:.2f}, {strat_name}={avg_score_s1:.2f} | Entropy: A1={entropy_a1:.2f}, {strat_name}={entropy_s1:.2f}')
        axs1[i].set_ylim(-0.1, 1.1)
        axs1[i].set_ylabel('Action (1=C, 0=D)')
        axs1[i].legend()

        # Plot agent2 vs strategy
        axs2[i].step(range(1, rounds + 1), moves_a2, label='Agent2 (RNN)')
        axs2[i].step(range(1, rounds + 1), moves_s2, label=strat_name, linestyle='dotted')
        axs2[i].set_title(f'Agent2 vs {strat_name}\nAvg: A2={avg_score_a2:.2f}, {strat_name}={avg_score_s2:.2f} | Entropy: A2={entropy_a2:.2f}, {strat_name}={entropy_s2:.2f}')
        axs2[i].set_ylim(-0.1, 1.1)
        axs2[i].set_ylabel('Action (1=C, 0=D)')
        axs2[i].legend()

    # Add common x-label
    fig1.text(0.5, 0.04, 'Round', ha='center', va='center')
    fig2.text(0.5, 0.04, 'Round', ha='center', va='center')
    
    fig1.tight_layout()
    fig2.tight_layout()
    
    fig1.savefig('agent1_vs_strategies.png')
    fig2.savefig('agent2_vs_strategies.png')
    
    plt.show()


def save_models(agent1, agent2):
    """
    Save trained agent models to disk.
    
    Args:
        agent1: First RL agent (MLP-based)
        agent2: Second RL agent (RNN-based)
    """
    torch.save(agent1.policy_net.state_dict(), 'agent1_mlp.pth')
    
    # For RNN agent, save both rnn and output layers
    torch.save({
        'rnn': agent2.rnn.state_dict(),
        'output': agent2.output.state_dict()
    }, 'agent2_rnn.pth')
    
    print("Models saved as 'agent1_mlp.pth' and 'agent2_rnn.pth'")


if __name__ == '__main__':
    # Initialize agents
    agent1 = MLPAgent(lr=0.01, entropy_weight=0.01)  # MLP agent using last 4 moves with entropy regularization
    agent2 = RNNAgent(lr=0.01, entropy_weight=0.01)  # RNN agent with hidden state and entropy regularization
    
    # Initialize strategies with more non-forgiving ones
    strategies = [
        AlwaysDefect(),
        AlwaysCooperate(),
        GrimTrigger(),
        TitForTat(),
        SuspiciousTitForTat(),
        Pavlov(),
        MajorityRule()
    ]

    # Train agents
    print("Training agents against multiple strategies...")
    performance_history = train_agents(agent1, agent2, strategies, episodes=10000)
    
    # Evaluate against fixed strategies and plot results
    print("Evaluating agents against fixed strategies...")
    evaluate_and_plot(agent1, agent2, strategies)
    
    # Save trained models
    save_models(agent1, agent2)
