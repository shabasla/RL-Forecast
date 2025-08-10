"""Soft Actor-Critic (SAC) agent implementation for discrete actions."""

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Tuple, Optional

from .networks import CNNLSTMNetwork
from ..memory.replay_buffer import PrioritizedReplayBuffer


class SACAgent:
    """Soft Actor-Critic agent for discrete action spaces.
    
    Implements SAC algorithm with:
    - Actor network for policy
    - Twin critic networks for Q-value estimation
    - Target critic networks for stability
    - Automatic temperature tuning
    
    Args:
        state_dim: Dimension of state space
        action_dim: Number of discrete actions
        sequence_length: Length of input sequences
        hidden_dim: Hidden dimension for networks
        lr: Learning rate
        gamma: Discount factor
        tau: Soft update parameter
        action_values: Array of action values
        target_entropy_factor: Target entropy scaling factor
        initial_log_alpha: Initial log temperature parameter
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int, 
        sequence_length: int,
        hidden_dim: int,
        lr: float,
        gamma: float,
        tau: float,
        action_values: np.ndarray,
        target_entropy_factor: float,
        initial_log_alpha: float,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_values = action_values
        self.gamma = gamma
        self.tau = tau
        self.action_dim = action_dim
        
        # Actor network
        self.actor = CNNLSTMNetwork(state_dim, action_dim, sequence_length, hidden_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        
        # Twin critic networks
        self.critic_1 = CNNLSTMNetwork(state_dim, action_dim, sequence_length, hidden_dim).to(self.device)
        self.critic_2 = CNNLSTMNetwork(state_dim, action_dim, sequence_length, hidden_dim).to(self.device)
        
        # Target critic networks
        self.target_critic_1 = CNNLSTMNetwork(state_dim, action_dim, sequence_length, hidden_dim).to(self.device)
        self.target_critic_2 = CNNLSTMNetwork(state_dim, action_dim, sequence_length, hidden_dim).to(self.device)
        
        # Initialize target networks
        self._initialize_target_networks()
        
        # Critic optimizer
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()), lr=lr
        )
        
        # Automatic temperature tuning
        self.target_entropy = -np.log(1.0 / action_dim) * target_entropy_factor
        self.log_alpha = torch.tensor(initial_log_alpha, requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
        
    def _initialize_target_networks(self) -> None:
        """Initialize target networks with main network parameters."""
        for target_param, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(param.data)
            
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[float, int]:
        """Select action using the current policy.
        
        Args:
            state: Current state observation
            deterministic: Whether to use deterministic policy
            
        Returns:
            Tuple of (action_value, action_index)
        """
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        logits = self.actor(state)
        
        if deterministic:
            action_idx = torch.argmax(logits, dim=1).item()
        else:
            dist = Categorical(logits=logits)
            action_idx = dist.sample().item()
            
        action = self.action_values[action_idx]
        return action, action_idx
        
    def update(
        self, 
        replay_buffer: PrioritizedReplayBuffer, 
        batch_size: int, 
        beta: float
    ) -> None:
        """Update agent networks using samples from replay buffer.
        
        Args:
            replay_buffer: Prioritized experience replay buffer
            batch_size: Number of samples to use for update
            beta: Importance sampling exponent
        """
        # Sample from replay buffer
        states, actions, rewards, next_states, dones, indices, weights = replay_buffer.sample(
            batch_size, beta
        )
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device).unsqueeze(1)
        weights = torch.FloatTensor(weights).to(self.device).unsqueeze(1)
        
        # Update critics
        self._update_critics(states, actions, rewards, next_states, dones, weights, replay_buffer, indices)
        
        # Update actor
        self._update_actor(states)
        
        # Update temperature parameter
        self._update_alpha(states)
        
        # Update target networks
        self._update_target_networks()
        
    def _update_critics(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        weights: torch.Tensor,
        replay_buffer: PrioritizedReplayBuffer,
        indices: np.ndarray,
    ) -> None:
        """Update critic networks."""
        with torch.no_grad():
            # Get next state action probabilities and Q-values
            next_logits = self.actor(next_states)
            next_dist = Categorical(logits=next_logits)
            next_probs = next_dist.probs
            
            next_q1 = self.target_critic_1(next_states)
            next_q2 = self.target_critic_2(next_states)
            next_q = torch.min(next_q1, next_q2)
            
            # Compute soft Q-target
            next_q = (next_probs * (next_q - self.log_alpha.exp() * torch.log(next_probs + 1e-8))).sum(
                dim=1, keepdim=True
            )
            target_q = rewards + (1 - dones) * self.gamma * next_q
            
        # Current Q-values
        current_q1 = self.critic_1(states).gather(1, actions.unsqueeze(1))
        current_q2 = self.critic_2(states).gather(1, actions.unsqueeze(1))
        
        # Critic loss with importance sampling weights
        critic_loss = (weights * (F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q))).mean()
        
        # Update critics
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update replay buffer priorities
        td_errors = torch.abs(current_q1 - target_q).detach().cpu().numpy().flatten()
        replay_buffer.update_priorities(indices, td_errors)
        
    def _update_actor(self, states: torch.Tensor) -> None:
        """Update actor network."""
        logits = self.actor(states)
        dist = Categorical(logits=logits)
        probs = dist.probs
        
        q1 = self.critic_1(states)
        q2 = self.critic_2(states)
        q_values = torch.min(q1, q2)
        
        # Actor loss (policy gradient with entropy regularization)
        actor_loss = (probs * (self.log_alpha.exp() * torch.log(probs + 1e-8) - q_values)).sum(dim=1).mean()
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
    def _update_alpha(self, states: torch.Tensor) -> None:
        """Update temperature parameter."""
        logits = self.actor(states)
        dist = Categorical(logits=logits)
        probs = dist.probs
        
        # Alpha loss (automatic temperature tuning)
        alpha_loss = -(self.log_alpha * (torch.log(probs + 1e-8) + self.target_entropy).detach()).mean()
        
        # Update alpha
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
    def _update_target_networks(self) -> None:
        """Soft update target networks."""
        for target_param, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
    @property
    def alpha(self) -> float:
        """Current temperature parameter value."""
        return self.log_alpha.exp().item()
        
    def save(self, path: str) -> None:
        """Save agent networks."""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic_1': self.critic_1.state_dict(),
            'critic_2': self.critic_2.state_dict(),
            'target_critic_1': self.target_critic_1.state_dict(),
            'target_critic_2': self.target_critic_2.state_dict(),
            'log_alpha': self.log_alpha.item(),
        }, path)
        
    def load(self, path: str) -> None:
        """Load agent networks."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic_1.load_state_dict(checkpoint['critic_1'])
        self.critic_2.load_state_dict(checkpoint['critic_2'])
        self.target_critic_1.load_state_dict(checkpoint['target_critic_1'])
        self.target_critic_2.load_state_dict(checkpoint['target_critic_2'])
        self.log_alpha = torch.tensor(checkpoint['log_alpha'], requires_grad=True, device=self.device)