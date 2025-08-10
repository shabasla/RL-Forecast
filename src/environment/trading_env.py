"""Trading environment implementation."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime


class TradingEnvironment:
    """Trading environment for reinforcement learning.
    
    Simulates a trading environment where an agent can adjust take-profit
    and stop-loss levels dynamically during trades.
    
    Args:
        data_file: Path to market data CSV file
        entry_points_file: Path to entry points CSV file
        state_cols: List of column names to use for state representation
        max_steps: Maximum number of steps per trade
        lookback_window: Number of historical observations in state
        ema_alpha: Alpha parameter for EMA normalization
        initial_tp: Initial take profit percentage
        initial_sl: Initial stop loss percentage
        tp_sl_width: Fixed width around center displacement for TP/SL
    """
    
    def __init__(
        self,
        data_file: str,
        entry_points_file: str,
        state_cols: List[str],
        max_steps: int,
        lookback_window: int,
        ema_alpha: float,
        initial_tp: float,
        initial_sl: float,
        tp_sl_width: float,
    ):
        # Load and prepare data
        self.data = pd.read_csv(data_file)
        self.entry_points = pd.read_csv(entry_points_file)
        self.state_cols = state_cols
        self.max_steps = max_steps
        self.lookback_window = lookback_window
        self.initial_tp = initial_tp
        self.initial_sl = initial_sl
        self.tp_sl_width = tp_sl_width
        
        # Convert dates and sort data
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.entry_points['date'] = pd.to_datetime(self.entry_points['date'])
        self.data = self.data.sort_values('date').reset_index(drop=True)
        self.entry_points = self.entry_points.sort_values('date').reset_index(drop=True)
        
        # Initialize EMA normalization parameters
        self.ema_means = {col: 0.0 for col in self.state_cols}
        self.ema_vars = {col: 1.0 for col in self.state_cols}
        self.alpha_norm = ema_alpha
        
        # Episode tracking
        self.total_entry_points = len(self.entry_points)
        self.reset()
        
        print(f"Environment initialized with {len(self.data)} data points and {self.total_entry_points} entry points")
        
    def reset(self) -> np.ndarray:
        """Reset environment for new episode.
        
        Returns:
            Initial state observation
        """
        # Start from first entry point
        self.current_entry_idx = 0
        self.episode_trades = []
        self.episode_reward = 0
        self.trades_completed = 0
        
        # Initialize first trade
        self._start_new_trade()
        
        print(f"Episode reset: Starting with {self.total_entry_points} entry points")
        
        return self._get_state()
        
    def _start_new_trade(self) -> None:
        """Initialize a new trade at current entry point."""
        if self.current_entry_idx >= self.total_entry_points:
            self.position_open = False
            return
            
        # Get current entry point
        entry_point = self.entry_points.iloc[self.current_entry_idx]
        self.entry_date = entry_point['date']
        
        # Find entry date in market data
        entry_data_idx = self.data[self.data['date'] == self.entry_date].index
        if len(entry_data_idx) == 0:
            # Find closest date
            entry_data_idx = self.data[self.data['date'] <= self.entry_date].index
            if len(entry_data_idx) == 0:
                # Skip this entry point
                self.current_entry_idx += 1
                if self.current_entry_idx < self.total_entry_points:
                    self._start_new_trade()
                else:
                    self.position_open = False
                return
            entry_data_idx = entry_data_idx[-1]
        else:
            entry_data_idx = entry_data_idx[0]
            
        self.entry_data_idx = entry_data_idx
        self.entry_price = self.data.iloc[entry_data_idx]['close']
        
        # Initialize position parameters (reset for each new trade)
        self.tp = self.initial_tp
        self.sl = self.initial_sl
        self.current_step = 0
        self.current_data_idx = entry_data_idx
        self.position_open = True
        
        print(f"  Trade {self.trades_completed + 1}: Entry date {self.entry_date}, Entry price {self.entry_price:.4f}")
        
    def _get_state(self) -> np.ndarray:
        """Get current state representation.
        
        Returns:
            Normalized state array of shape (lookback_window, n_features)
        """
        if not self.position_open:
            return np.zeros((self.lookback_window, len(self.state_cols)))
            
        # Get lookback window data ending at current_data_idx
        end_idx = self.current_data_idx
        start_idx = max(0, end_idx - self.lookback_window + 1)
        
        # Extract state data
        state_data = self.data.iloc[start_idx:end_idx + 1][self.state_cols].values
        
        # Pad if insufficient data
        if len(state_data) < self.lookback_window:
            padded_state = np.zeros((self.lookback_window, len(self.state_cols)))
            padded_state[-len(state_data):] = state_data
            state_data = padded_state
        
        # Apply EMA-based z-score normalization
        normalized_state = np.zeros_like(state_data)
        for i, col in enumerate(self.state_cols):
            for t in range(state_data.shape[0]):
                value = state_data[t, i]
                if not np.isnan(value):
                    # Update EMA statistics
                    self.ema_means[col] = (1 - self.alpha_norm) * self.ema_means[col] + self.alpha_norm * value
                    self.ema_vars[col] = (1 - self.alpha_norm) * self.ema_vars[col] + self.alpha_norm * (value - self.ema_means[col])**2
                    
                    # Normalize
                    normalized_value = (value - self.ema_means[col]) / (np.sqrt(self.ema_vars[col]) + 1e-8)
                    normalized_state[t, i] = normalized_value
                    
        return normalized_state
        
    def step(self, action: float) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Take a step in the environment.
        
        Args:
            action: Action value (center displacement for TP/SL)
            
        Returns:
            Tuple containing:
                - next_state: Next state observation
                - reward: Reward for this step
                - done: Whether episode is complete
                - info: Additional information dictionary
        """
        if not self.position_open:
            return (
                np.zeros((self.lookback_window, len(self.state_cols))), 
                0, 
                True, 
                {
                    'trades_completed': self.trades_completed,
                    'episode_trades': self.episode_trades,
                    'reason': 'all_trades_completed'
                }
            )

        self.current_step += 1
        next_data_idx = self.entry_data_idx + self.current_step
        
        # Check if we've run out of data
        if next_data_idx >= len(self.data):
            current_price = self.data.iloc[self.current_data_idx]['close']
            ret = (current_price - self.entry_price) / self.entry_price
            reward = self._calculate_reward(ret, tp_sl_hit=False)
            self._close_trade(ret, 'no_more_data')
            return self._get_next_state_or_episode_end(reward)
        
        self.current_data_idx = next_data_idx
        current_price = self.data.iloc[self.current_data_idx]['close']
        ret = (current_price - self.entry_price) / self.entry_price
        
        # Check for TP/SL hits or max steps
        trade_done, tp_sl_hit, reason = False, False, 'continuing'
        
        if ret >= self.tp:
            trade_done, tp_sl_hit, reason = True, True, 'tp_hit'
            print(f"    TP hit! Return: {ret:.4f}, TP: {self.tp:.4f}")
        elif ret <= self.sl:
            trade_done, tp_sl_hit, reason = True, True, 'sl_hit'
            print(f"    SL hit! Return: {ret:.4f}, SL: {self.sl:.4f}")
        elif self.current_step >= self.max_steps:
            trade_done, reason = True, 'max_steps'
            print(f"    Max steps reached! Return: {ret:.4f}")
            
        if trade_done:
            reward = self._calculate_reward(ret, tp_sl_hit)
            self._close_trade(ret, reason)
            return self._get_next_state_or_episode_end(reward)
        else:
            # Continue trade - update TP/SL based on action
            reward = self._calculate_reward(ret, tp_sl_hit=False)
            center_displacement = action
            self.tp = center_displacement + self.tp_sl_width
            self.sl = center_displacement - self.tp_sl_width
            
            next_state = self._get_state()
            return next_state, reward, False, {
                'return': ret, 
                'reason': reason, 
                'tp': self.tp, 
                'sl': self.sl,
                'trade_num': self.trades_completed + 1, 
                'step': self.current_step
            }
            
    def _close_trade(self, final_return: float, reason: str) -> None:
        """Close current trade and record results.
        
        Args:
            final_return: Final return for the trade
            reason: Reason for trade closure
        """
        trade_info = {   
            'entry_date': self.entry_date,
            'entry_price': self.entry_price,
            'final_return': final_return,
            'steps': self.current_step,
            'reason': reason
        }
        self.episode_trades.append(trade_info)
        self.trades_completed += 1
        
        print(f"    Trade {self.trades_completed} completed: Return {final_return:.4f}, Steps {self.current_step}, Reason: {reason}")
        
        # Move to next entry point
        self.current_entry_idx += 1
        self._start_new_trade()
        
    def _get_next_state_or_episode_end(self, reward: float) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Get next state or signal episode end.
        
        Args:
            reward: Current step reward
            
        Returns:
            Tuple of (state, reward, done, info)
        """
        episode_done = not self.position_open
        
        if episode_done:
            total_return = sum(trade['final_return'] for trade in self.episode_trades)
            avg_return = total_return / len(self.episode_trades) if self.episode_trades else 0
            
            next_state = np.zeros((self.lookback_window, len(self.state_cols)))
            return next_state, reward, True, {
                'trades_completed': self.trades_completed,
                'total_return': total_return,
                'avg_return': avg_return,
                'episode_trades': self.episode_trades,
                'reason': 'all_trades_completed'
            }
        else:
            next_state = self._get_state()
            return next_state, reward, False, {
                'trade_num': self.trades_completed,
                'trades_completed': self.trades_completed,
                'continuing_to_next_trade': True
            }
        
    def _calculate_reward(self, ret: float, tp_sl_hit: bool) -> float:
        """Calculate reward for current step.
        
        Implements shaped reward function:
        - Exponential reward for positive returns
        - Cubic penalty for negative returns  
        - Bonus when TP/SL is hit
        
        Args:
            ret: Current return
            tp_sl_hit: Whether TP or SL was hit
            
        Returns:
            Calculated reward value
        """
        if ret > 0:
            shaped = np.exp(ret)
        elif ret == 0:  
            shaped = 0
        else:  # ret < 0
            shaped = ret * ret * ret

        bonus = ret if tp_sl_hit else 0
        return shaped + bonus