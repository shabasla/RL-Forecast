#Dual Agent Trading System with Long/Short Position Management
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import os
from collections import deque
import random
from datetime import datetime, timedelta

# ==============================================================================
#                 CONFIGURATION & HYPERPARAMETERS
# ==============================================================================

# --- File Paths ---
# NOTE: Update these paths to point to your local files.
TEST_DATA_FILE = "test_data18alt.csv"
TEST_ENTRY_POINTS_FILE = "deflection_points(18).csv"  # Must contain 'date' and 'signal' columns
LONG_ACTOR_WEIGHTS_FILE = "final_actor_weights(longy).pth"  # Path to long position weights
SHORT_ACTOR_WEIGHTS_FILE = "final_actor_weights(shorty).pth"  # Path to short position weights

# --- Environment Parameters ---
MAX_TRADE_STEPS = 7
LOOKBACK_WINDOW = 15 # This should be the same as SEQUENCE_LENGTH
STATE_COLS = ['sto_osc', 'macd', 'adx', 'obv', 'n_atr', 'log_ret', 'newsapi']
EMA_ALPHA_NORM = 0.05 # Alpha for EMA normalization of state features
INITIAL_TP = 0.03      # Initial Take Profit percentage (3%)
INITIAL_SL = -0.03     # Initial Stop Loss percentage (-3%)
TP_SL_WIDTH = 0.03     # The fixed distance from the center displacement for TP/SL

# --- Trading Cost Parameters ---
SLIPPAGE_BPS = 5           # Slippage in basis points (5 bps = 0.05%)
TRANSACTION_COST_PCT = 0.0015 # Transaction cost as a percentage (0.15%)

# --- Model & Agent Hyperparameters ---
SEQUENCE_LENGTH = 15 # This should be the same as LOOKBACK_WINDOW
HIDDEN_DIM = 128
ACTION_VALUES = np.array([-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7]) / 100
ACTION_DIM = len(ACTION_VALUES)
LEARNING_RATE = 1e-4
GAMMA = 0.99           # Discount factor for future rewards
TAU = 0.005            # Soft update factor for target networks
TARGET_ENTROPY_FACTOR = 0.98 # Target entropy scaling factor
INITIAL_LOG_ALPHA = np.log(0.2) # Initial value for the temperature parameter alpha

# --- Replay Buffer Hyperparameters (for training, not used in this test script) ---
BUFFER_CAPACITY = 100000
PER_ALPHA = 0.6
PER_BETA_START = 0.4
PER_BETA_END = 1.0

# --- Testing Script Parameters ---
INITIAL_PORTFOLIO_VALUE = 10000.0

# ==============================================================================
#                 END OF CONFIGURATION
# ==============================================================================


# Replay Buffer with PER
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = deque(maxlen=capacity)
        self.pos = 0

    def push(self, state, action, reward, next_state, done):
        max_priority = max(self.priorities) if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.priorities.append(max_priority)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        priorities = np.array(self.priorities, dtype=np.float32)
        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()

        states, actions, rewards, next_states, dones = zip(*samples)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones), indices, weights)

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            if idx < len(self.priorities):
                self.priorities[idx] = float(priority) + 1e-5

    def __len__(self):
        return len(self.buffer)

# CNN-LSTM Network
class CNNLSTMNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, sequence_length, hidden_dim):
        super(CNNLSTMNetwork, self).__init__()

        self.conv1 = nn.Conv1d(state_dim, 32, kernel_size=3, padding=1)
        self.ln1 = nn.LayerNorm([32, sequence_length])
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.ln2 = nn.LayerNorm([64, sequence_length])

        self.lstm = nn.LSTM(64, hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        # x shape: [batch, sequence, features]
        x = x.permute(0, 2, 1)  # [batch, features, sequence]
        x = F.relu(self.ln1(self.conv1(x)))
        x = F.relu(self.ln2(self.conv2(x)))
        x = x.permute(0, 2, 1)  # [batch, sequence, channels]
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Take last LSTM output
        x = F.relu(self.ln3(self.fc1(x)))
        return self.fc2(x)

# SAC Agent with Discrete Actions
class SACAgent:
    def __init__(self, state_dim, action_dim, sequence_length, hidden_dim, lr, gamma, tau, action_values, target_entropy_factor, initial_log_alpha):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_values = action_values

        # Actor
        self.actor = CNNLSTMNetwork(state_dim, action_dim, sequence_length, hidden_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        # Critics
        self.critic_1 = CNNLSTMNetwork(state_dim, action_dim, sequence_length, hidden_dim).to(self.device)
        self.critic_2 = CNNLSTMNetwork(state_dim, action_dim, sequence_length, hidden_dim).to(self.device)
        self.target_critic_1 = CNNLSTMNetwork(state_dim, action_dim, sequence_length, hidden_dim).to(self.device)
        self.target_critic_2 = CNNLSTMNetwork(state_dim, action_dim, sequence_length, hidden_dim).to(self.device)

        # Initialize target networks
        for target_param, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(param.data)

        self.critic_optimizer = torch.optim.Adam(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()), lr=lr)

        self.gamma = gamma
        self.tau = tau
        self.action_dim = action_dim

        # Log entropy target
        self.target_entropy = -np.log(1.0 / action_dim) * target_entropy_factor
        self.log_alpha = torch.tensor(initial_log_alpha, requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

    def select_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        logits = self.actor(state)
        probs = torch.softmax(logits, dim=1).squeeze().detach().cpu().numpy()

        pos_indices = [i for i, val in enumerate(self.action_values) if val > 0]
        zero_indices = [i for i, val in enumerate(self.action_values) if val == 0]
        neg_indices = [i for i, val in enumerate(self.action_values) if val < 0]

        pos_prob = sum(probs[i] for i in pos_indices)
        zero_prob = sum(probs[i] for i in zero_indices)
        neg_prob = sum(probs[i] for i in neg_indices)

        class_probs = [pos_prob, zero_prob, neg_prob]
        class_indices = [pos_indices, zero_indices, neg_indices]

        # Deterministic class selection
        chosen_class_idx = np.argmax(class_probs)  # Always choose the class with the highest probability

        chosen_indices = class_indices[chosen_class_idx]
        if not chosen_indices:
          action_idx = np.argmax(probs)  # Fallback if chosen class is empty
        else:
    # Stochastic sampling within the chosen class
         within_class_probs = np.array([probs[i] for i in chosen_indices])
         within_class_probs /= within_class_probs.sum()  # Normalize probabilities
         action_idx = np.random.choice(chosen_indices, p=within_class_probs)

        action = self.action_values[action_idx]
        return action, action_idx

# DUAL AGENT TRADING ENVIRONMENT WITH LONG/SHORT SUPPORT
class DualAgentTradingEnvironment:
    def __init__(self, config):
        self.data = pd.read_csv(config['data_file'])
        self.entry_points = pd.read_csv(config['entry_points_file'])
        self.state_cols = config['state_cols']
        self.max_steps = config['max_steps']
        self.lookback_window = config['lookback_window']
        self.initial_tp = config['initial_tp']
        self.initial_sl = config['initial_sl']
        self.tp_sl_width = config['tp_sl_width']
        self.alpha_norm = config['ema_alpha_norm']
        self.slippage_bps = config['slippage_bps']
        self.transaction_cost_pct = config['transaction_cost_pct']

        self.data['date'] = pd.to_datetime(self.data['date'])
        self.entry_points['date'] = pd.to_datetime(self.entry_points['date'])

        self.data = self.data.sort_values('date').reset_index(drop=True)
        self.entry_points = self.entry_points.sort_values('date').reset_index(drop=True)

        if 'signal' not in self.entry_points.columns:
            raise ValueError("Entry points file must contain 'signal' column with values -1 (long) or 1 (short)")

        self.ema_means = {col: 0 for col in self.state_cols}
        self.ema_vars = {col: 1 for col in self.state_cols}

        self.delayed_exit = False
        self.pending_reward = 0
        self.pending_reason = None
        self.pending_return = 0

        self.total_entry_points = len(self.entry_points)
        print(f"Environment initialized with {len(self.data)} data points and {self.total_entry_points} entry points")

    def reset(self):
        self.last_exit_date = pd.Timestamp.min
        self.episode_trades = []
        self.episode_reward = 0
        self.trades_completed = 0
        self.delayed_exit = False
        self.pending_reward = 0
        self.pending_reason = None
        self.pending_return = 0

        self._start_new_trade()
        print(f"Episode reset: Starting with {self.total_entry_points} entry points, following time.")
        return self._get_state()

    def _start_new_trade(self):
        valid_entry_points = self.entry_points[self.entry_points['date'] > self.last_exit_date]

        if valid_entry_points.empty:
            self.position_open = False
            return

        entry_point = valid_entry_points.iloc[0]
        self.entry_date = entry_point['date']
        self.signal = entry_point['signal']

        entry_data_idx = self.data[self.data['date'] >= self.entry_date].index
        if len(entry_data_idx) == 0:
            self.position_open = False
            return
        entry_data_idx = entry_data_idx[0]

        self.entry_data_idx = entry_data_idx
        self.entry_price = self.data.iloc[entry_data_idx]['close']

        slippage_factor = self.slippage_bps / 10000.0
        if self.signal == -1:
            self.entry_price_with_slippage = self.entry_price * (1 + slippage_factor)
        else:
            self.entry_price_with_slippage = self.entry_price * (1 - slippage_factor)

        self.tp = self.initial_tp
        self.sl = self.initial_sl
        self.current_step = 0
        self.current_data_idx = entry_data_idx
        self.position_open = True

        position_type = "LONG" if self.signal == -1 else "SHORT"
        print(f"  Trade {self.trades_completed + 1}: {position_type} position, Entry date {self.entry_date}, Entry price {self.entry_price:.4f} (with slippage: {self.entry_price_with_slippage:.4f})")

    def _get_state(self):
        if not self.position_open:
            return np.zeros((self.lookback_window, len(self.state_cols)))

        end_idx = self.current_data_idx
        start_idx = max(0, end_idx - self.lookback_window + 1)
        state_data = self.data.iloc[start_idx:end_idx + 1][self.state_cols].values

        if len(state_data) < self.lookback_window:
            padded_state = np.zeros((self.lookback_window, len(self.state_cols)))
            padded_state[-len(state_data):] = state_data
            state_data = padded_state

        normalized_state = np.zeros_like(state_data)
        for i, col in enumerate(self.state_cols):
            for t in range(state_data.shape[0]):
                value = state_data[t, i]
                if not np.isnan(value):
                    self.ema_means[col] = (1 - self.alpha_norm) * self.ema_means[col] + self.alpha_norm * value
                    self.ema_vars[col] = (1 - self.alpha_norm) * self.ema_vars[col] + self.alpha_norm * (value - self.ema_means[col])**2
                    normalized_value = (value - self.ema_means[col]) / (np.sqrt(self.ema_vars[col]) + 1e-8)
                    normalized_state[t, i] = normalized_value

        return normalized_state

    def step(self, action):
        if self.delayed_exit:
            self._close_trade(self.pending_return, self.pending_reason)
            self.delayed_exit = False
            return self._get_next_state_or_episode_end(self.pending_reward)

        if not self.position_open:
            return np.zeros((self.lookback_window, len(self.state_cols))), 0, True, {'trades_completed': self.trades_completed, 'episode_trades': self.episode_trades, 'reason': 'all_trades_completed'}

        action_value = action
        self.current_step += 1
        next_data_idx = self.entry_data_idx + self.current_step

        if next_data_idx >= len(self.data):
            current_price = self.data.iloc[self.current_data_idx]['close']
            ret = self._calculate_position_return(current_price)
            reward = self._calculate_reward(ret, tp_sl_hit=False)
            self._close_trade(ret, 'no_more_data')
            return self._get_next_state_or_episode_end(reward)

        self.current_data_idx = next_data_idx
        current_price = self.data.iloc[self.current_data_idx]['close']
        ret = self._calculate_position_return(current_price)

        trade_done, tp_sl_hit, reason = False, False, 'continuing'

        if ret >= self.tp:
            self.delayed_exit = True
            self.pending_return = ret
            self.pending_reason = 'tp_hit'
            print(f"    TP hit (delayed exit)! Return: {ret:.4f}, TP: {self.tp:.4f}")
            return self._get_state(), 0, False, {'return': ret, 'tp': self.tp, 'sl': self.sl, 'step': self.current_step, 'delayed_exit': True, 'position_type': 'LONG' if self.signal == -1 else 'SHORT'}
        elif ret <= self.sl:
            self.delayed_exit = True
            self.pending_return = ret
            self.pending_reason = 'sl_hit'
            print(f"    SL hit (delayed exit)! Return: {ret:.4f}, SL: {self.sl:.4f}")
            return self._get_state(), 0, False, {'return': ret, 'tp': self.tp, 'sl': self.sl, 'step': self.current_step, 'delayed_exit': True, 'position_type': 'LONG' if self.signal == -1 else 'SHORT'}
        elif self.current_step >= self.max_steps:
            trade_done, reason = True, 'max_steps'
            print(f"    Max steps reached! Return: {ret:.4f}")

        reward = self._calculate_reward(ret, tp_sl_hit)

        if trade_done:
            self._close_trade(ret, reason)
            return self._get_next_state_or_episode_end(reward)
        else:
            center_displacement = action_value
            self.tp = center_displacement + self.tp_sl_width
            self.sl = center_displacement - self.tp_sl_width
            next_state = self._get_state()
            print(f"    Current BTC Price: {current_price:.4f}") # Added line
            return next_state, reward, False, {'return': ret, 'reason': reason, 'tp': self.tp, 'sl': self.sl, 'trade_num': self.trades_completed + 1, 'step': self.current_step, 'position_type': 'LONG' if self.signal == -1 else 'SHORT'}

    def _calculate_position_return(self, current_price):
        #if self.signal == -1:
        ret = (current_price - self.entry_price_with_slippage) / self.entry_price_with_slippage
        #else:
            #ret = (self.entry_price_with_slippage - current_price) / self.entry_price_with_slippage
        return ret

    def _close_trade(self, final_return, reason):
        if self.signal == -1:
           final_return_after_costs = final_return - self.transaction_cost_pct
        else:
           final_return_after_costs = -final_return - self.transaction_cost_pct

        trade_info = {
            'entry_date': self.entry_date, 'entry_price': self.entry_price, 'entry_price_with_slippage': self.entry_price_with_slippage,
            'signal': self.signal, 'position_type': 'LONG' if self.signal == -1 else 'SHORT', 'final_return': final_return,
            'final_return_after_costs': final_return_after_costs, 'transaction_cost': self.transaction_cost_pct,
            'slippage_cost': abs(self.entry_price_with_slippage - self.entry_price) / self.entry_price, 'steps': self.current_step, 'reason': reason
        }
        self.episode_trades.append(trade_info)
        self.trades_completed += 1
        position_type = "LONG" if self.signal == -1 else "SHORT"
        print(f"    Trade {self.trades_completed} completed: {position_type} Return {final_return:.4f} (After costs: {final_return_after_costs:.4f}), Steps {self.current_step}, Reason: {reason}")

        self.last_exit_date = self.data.iloc[self.current_data_idx]['date']
        self._start_new_trade()

    def _get_next_state_or_episode_end(self, reward):
        episode_done = not self.position_open
        if episode_done:
            total_return = sum(trade['final_return_after_costs'] for trade in self.episode_trades)
            avg_return = total_return / len(self.episode_trades) if self.episode_trades else 0
            next_state = np.zeros((self.lookback_window, len(self.state_cols)))
            return next_state, reward, True, {'trades_completed': self.trades_completed, 'total_return': total_return, 'avg_return': avg_return, 'episode_trades': self.episode_trades, 'reason': 'all_trades_completed'}
        else:
            next_state = self._get_state()
            return next_state, reward, False, {'trade_num': self.trades_completed + 1, 'trades_completed': self.trades_completed, 'continuing_to_next_trade': True}

    def _calculate_reward(self, ret, tp_sl_hit):
        reward = np.exp(ret/10) - 1 + ret
        return reward

# DUAL AGENT MANAGER
class DualAgentManager:
    def __init__(self, config):
        self.long_agent = SACAgent(
            state_dim=len(config['state_cols']), action_dim=config['action_dim'], sequence_length=config['sequence_length'],
            hidden_dim=config['hidden_dim'], lr=config['learning_rate'], gamma=config['gamma'], tau=config['tau'],
            action_values=config['action_values'], target_entropy_factor=config['target_entropy_factor'],
            initial_log_alpha=config['initial_log_alpha']
        )
        self.short_agent = SACAgent(
            state_dim=len(config['state_cols']), action_dim=config['action_dim'], sequence_length=config['sequence_length'],
            hidden_dim=config['hidden_dim'], lr=config['learning_rate'], gamma=config['gamma'], tau=config['tau'],
            action_values=config['action_values'], target_entropy_factor=config['target_entropy_factor'],
            initial_log_alpha=config['initial_log_alpha']
        )

    def load_weights(self, long_weights_path, short_weights_path):
        try:
            self.long_agent.actor.load_state_dict(torch.load(long_weights_path, map_location=self.long_agent.device))
            self.long_agent.actor.eval()
            print(f"Successfully loaded LONG agent weights from {long_weights_path}")

            self.short_agent.actor.load_state_dict(torch.load(short_weights_path, map_location=self.short_agent.device))
            self.short_agent.actor.eval()
            print(f"Successfully loaded SHORT agent weights from {short_weights_path}")

        except FileNotFoundError as e:
            print(f"Error: Weights file not found: {e}")
            raise
        except Exception as e:
            print(f"An error occurred while loading weights: {e}")
            raise

    def select_action(self, state, signal, deterministic=True):
        if signal == -1:
            return self.long_agent.select_action(state, deterministic)
        else:
            return self.short_agent.select_action(state, deterministic)

# ----- PERFORMANCE METRICS CALCULATION -----
def calculate_and_print_performance_metrics(initial_portfolio, trades):
    if not trades:
        print("\nNo trades were made, cannot calculate performance metrics.")
        return

    portfolio_value = initial_portfolio
    portfolio_history = [initial_portfolio]
    trade_returns = []
    long_trades = []
    short_trades = []

    for trade in trades:
        final_return = trade['final_return_after_costs']
        trade_returns.append(final_return)
        portfolio_value *= (1 + final_return)
        portfolio_history.append(portfolio_value)
        if trade['position_type'] == 'LONG':
            long_trades.append(trade)
        else:
            short_trades.append(trade)

    portfolio_history = np.array(portfolio_history)
    trade_returns = np.array(trade_returns)

    final_portfolio_value = portfolio_history[-1]
    total_return_pct = (final_portfolio_value - initial_portfolio) / initial_portfolio

    peaks = np.maximum.accumulate(portfolio_history)
    drawdowns = (peaks - portfolio_history) / peaks
    max_drawdown_pct = np.max(drawdowns)

    if len(trades) > 1:
        start_date = trades[0]['entry_date']
        end_date_last_trade = trades[-1]['entry_date'] + timedelta(days=trades[-1]['steps'])
        total_days = (end_date_last_trade - start_date).days
        annual_trading_freq = len(trades) / (total_days / 365.25) if total_days > 0 else 1.0
    else:
        annual_trading_freq = 1.0
        total_days = sum(t['steps'] for t in trades)

    if np.std(trade_returns) > 1e-8:
        sharpe_ratio = np.mean(trade_returns) / np.std(trade_returns)
        annualized_sharpe_ratio = sharpe_ratio * np.sqrt(annual_trading_freq)
    else:
        sharpe_ratio = float('inf') if np.mean(trade_returns) > 0 else 0
        annualized_sharpe_ratio = float('inf') if np.mean(trade_returns) > 0 else 0

    long_returns = [t['final_return_after_costs'] for t in long_trades] if long_trades else [0]
    short_returns = [t['final_return_after_costs'] for t in short_trades] if short_trades else []

    total_transaction_costs = sum(t['transaction_cost'] for t in trades)
    total_slippage_costs = sum(t['slippage_cost'] for t in trades)

    print("\n\n" + "=" * 80)
    print("DUAL AGENT PORTFOLIO PERFORMANCE METRICS")
    print("=" * 80)
    print(f"  Initial Portfolio Value: ${initial_portfolio:,.2f}")
    print(f"  Final Portfolio Value:   ${final_portfolio_value:,.2f}")
    print(f"  Total Portfolio Return:  {total_return_pct:.2%}")
    print(f"  Max Drawdown:            {max_drawdown_pct:.2%}")
    print(f"  Annualized Sharpe Ratio: {annualized_sharpe_ratio:.2f}")
    print()
    print("POSITION TYPE BREAKDOWN:")
    print(f"  Long Trades:   {len(long_trades)} trades, Avg Return: {np.mean(long_returns):.2%}")
    print(f"  Short Trades:  {len(short_trades)} trades, Avg Return: {np.mean(short_returns):.2%}")
    print()
    print("TRADING COSTS:")
    print(f"  Total Transaction Costs: {total_transaction_costs:.4f} ({total_transaction_costs/len(trades)*100:.2f}% per trade)")
    print(f"  Total Slippage Costs:    {total_slippage_costs:.4f} ({total_slippage_costs/len(trades)*100:.2f}% per trade)")
    print("=" * 80)

# ----- TESTING SCRIPT FOR DUAL AGENTS -----
def test_dual_sac(config):
    env = DualAgentTradingEnvironment(config['env_config'])
    dual_agent = DualAgentManager(config['agent_config'])
    dual_agent.load_weights(config['long_weights_path'], config['short_weights_path'])

    state = env.reset()
    done = False
    step_count = 0

    print("\nStarting Dual Agent Testing Loop...")
    print("=" * 100)

    while not done:
        current_signal = env.signal if env.position_open else 0
        action, action_idx = dual_agent.select_action(state, current_signal, deterministic=True)
        next_state, reward, done, info = env.step(action)
        state = next_state
        step_count += 1

        if not done and 'trade_num' in info:
            position_type = info.get('position_type', 'UNKNOWN')
            print(f"    Step {info.get('step', 0)}: {position_type} Action {action:.3f}, Return {info.get('return', 0):.4f}, TP {info.get('tp', 0):.3f}, SL {info.get('sl', 0):.3f}")

    print("\nDual Agent Testing Finished!")
    print("=" * 100)

    trades_completed = info.get('trades_completed', 0)
    all_trades = info.get('episode_trades', [])

    print("\nTesting Summary:")
    print(f"  Trades Completed: {trades_completed}")
    print(f"  Total Steps: {step_count}")

    if all_trades:
        results_df = pd.DataFrame(all_trades)
        results_df.to_csv("dual_agent_test_results.csv", index=False)
        print("\nTest results saved to dual_agent_test_results.csv")
        print(results_df[['entry_date', 'position_type', 'final_return', 'final_return_after_costs', 'reason']])
        calculate_and_print_performance_metrics(config['initial_portfolio'], all_trades)

if __name__ == "__main__":
    # Validate file existence before running
    required_files = [TEST_DATA_FILE, TEST_ENTRY_POINTS_FILE, LONG_ACTOR_WEIGHTS_FILE, SHORT_ACTOR_WEIGHTS_FILE]
    files_exist = True
    for f in required_files:
        if not os.path.exists(f):
            print(f"Error: Required file not found: {f}")
            files_exist = False

    if files_exist:
        # Pack all hyperparameters into configuration dictionaries
        env_config = {
            "data_file": TEST_DATA_FILE,
            "entry_points_file": TEST_ENTRY_POINTS_FILE,
            "max_steps": MAX_TRADE_STEPS,
            "lookback_window": LOOKBACK_WINDOW,
            "state_cols": STATE_COLS,
            "ema_alpha_norm": EMA_ALPHA_NORM,
            "initial_tp": INITIAL_TP,
            "initial_sl": INITIAL_SL,
            "tp_sl_width": TP_SL_WIDTH,
            "slippage_bps": SLIPPAGE_BPS,
            "transaction_cost_pct": TRANSACTION_COST_PCT
        }

        agent_config = {
            "state_cols": STATE_COLS,
            "action_dim": ACTION_DIM,
            "sequence_length": SEQUENCE_LENGTH,
            "hidden_dim": HIDDEN_DIM,
            "learning_rate": LEARNING_RATE,
            "gamma": GAMMA,
            "tau": TAU,
            "action_values": ACTION_VALUES,
            "target_entropy_factor": TARGET_ENTROPY_FACTOR,
            "initial_log_alpha": INITIAL_LOG_ALPHA,
        }

        full_config = {
            "env_config": env_config,
            "agent_config": agent_config,
            "long_weights_path": LONG_ACTOR_WEIGHTS_FILE,
            "short_weights_path": SHORT_ACTOR_WEIGHTS_FILE,
            "initial_portfolio": INITIAL_PORTFOLIO_VALUE
        }

        test_dual_sac(full_config)