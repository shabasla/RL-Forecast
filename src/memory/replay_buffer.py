"""Prioritized Experience Replay Buffer implementation."""

import numpy as np
from collections import deque
from typing import Tuple, List, Any


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay (PER) buffer.
    
    Implements importance sampling-based prioritized replay to improve
    sample efficiency by prioritizing experiences with higher TD errors.
    
    Args:
        capacity: Maximum buffer size
        alpha: Prioritization exponent (0 = uniform, 1 = fully prioritized)
    """
    
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = deque(maxlen=capacity)
        self.pos = 0
        
    def push(
        self, 
        state: np.ndarray, 
        action: int, 
        reward: float, 
        next_state: np.ndarray, 
        done: bool
    ) -> None:
        """Add a new experience to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        # Assign maximum priority to new experiences
        max_priority = max(self.priorities) if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
            
        self.priorities.append(max_priority)
        self.pos = (self.pos + 1) % self.capacity
        
    def sample(
        self, 
        batch_size: int, 
        beta: float = 0.4
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample a batch of experiences with importance sampling weights.
        
        Args:
            batch_size: Number of samples to draw
            beta: Importance sampling exponent (0 = no correction, 1 = full correction)
            
        Returns:
            Tuple containing:
                - states: Batch of states
                - actions: Batch of actions
                - rewards: Batch of rewards
                - next_states: Batch of next states
                - dones: Batch of done flags
                - indices: Indices of sampled experiences
                - weights: Importance sampling weights
        """
        # Convert priorities to probabilities
        priorities = np.array(self.priorities, dtype=np.float32)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices based on priorities
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()  # Normalize weights
        
        # Unpack samples
        states, actions, rewards, next_states, dones = zip(*samples)
        
        return (
            np.array(states), 
            np.array(actions), 
            np.array(rewards),
            np.array(next_states), 
            np.array(dones), 
            indices, 
            weights
        )
                
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """Update priorities for sampled experiences.
        
        Args:
            indices: Indices of experiences to update
            priorities: New priority values (typically TD errors)
        """
        for idx, priority in zip(indices, priorities):
            if idx < len(self.priorities):
                # Add small epsilon to avoid zero priorities
                self.priorities[idx] = float(priority) + 1e-5
            
    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)
        
    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples for training.
        
        Args:
            batch_size: Required batch size
            
        Returns:
            True if buffer has enough samples
        """
        return len(self.buffer) >= batch_size