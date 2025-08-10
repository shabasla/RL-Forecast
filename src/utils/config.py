"""Configuration utilities for the SAC trading agent."""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Union


class Config:
    """Configuration manager for SAC trading agent.
    
    Provides easy access to configuration parameters and validation.
    """
    
    def __init__(self, config_path: Union[str, Path, None] = None):
        """Initialize configuration.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config = {}
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: Union[str, Path]) -> None:
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        required_sections = [
            'data', 'environment', 'model', 'agent', 
            'replay_buffer', 'training', 'logging'
        ]
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate specific parameters
        env_config = self.config['environment']
        if env_config['lookback_window'] != self.config['model']['sequence_length']:
            raise ValueError("lookback_window must equal sequence_length")
        
        if len(self.config['model']['action_values']) == 0:
            raise ValueError("action_values cannot be empty")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to configuration value (e.g., 'agent.learning_rate')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def update(self, key_path: str, value: Any) -> None:
        """Update configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to configuration value
            value: New value to set
        """
        keys = key_path.split('.')
        config = self.config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
    
    def save(self, output_path: Union[str, Path]) -> None:
        """Save configuration to YAML file.
        
        Args:
            output_path: Path to save configuration file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
    
    @property
    def data_config(self) -> Dict[str, Any]:
        """Get data configuration section."""
        return self.config.get('data', {})
    
    @property
    def environment_config(self) -> Dict[str, Any]:
        """Get environment configuration section."""
        return self.config.get('environment', {})
    
    @property
    def model_config(self) -> Dict[str, Any]:
        """Get model configuration section."""
        return self.config.get('model', {})
    
    @property
    def agent_config(self) -> Dict[str, Any]:
        """Get agent configuration section."""
        return self.config.get('agent', {})
    
    @property
    def replay_buffer_config(self) -> Dict[str, Any]:
        """Get replay buffer configuration section."""
        return self.config.get('replay_buffer', {})
    
    @property
    def training_config(self) -> Dict[str, Any]:
        """Get training configuration section."""
        return self.config.get('training', {})
    
    @property
    def logging_config(self) -> Dict[str, Any]:
        """Get logging configuration section."""
        return self.config.get('logging', {})


def create_default_config() -> Dict[str, Any]:
    """Create default configuration dictionary.
    
    Returns:
        Default configuration dictionary
    """
    return {
        'data': {
            'training_data': 'data/trainalt_data.csv',
            'entry_points': 'data/deflection_data.csv',
            'model_save_path': 'models/'
        },
        'environment': {
            'max_trade_steps': 7,
            'lookback_window': 15,
            'state_columns': ['sto_osc', 'macd', 'adx', 'obv', 'n_atr', 'log_ret', 'newsapi'],
            'ema_alpha_norm': 0.05,
            'initial_tp': 0.03,
            'initial_sl': -0.03,
            'tp_sl_width': 0.03
        },
        'model': {
            'sequence_length': 15,
            'hidden_dim': 128,
            'action_values': [-0.07, -0.06, -0.05, -0.04, -0.03, -0.02, -0.01, 
                             0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
        },
        'agent': {
            'learning_rate': 0.0001,
            'gamma': 0.99,
            'tau': 0.005,
            'target_entropy_factor': 0.98,
            'initial_log_alpha': -1.6094379124341003
        },
        'replay_buffer': {
            'capacity': 100000,
            'per_alpha': 0.6,
            'per_beta_start': 0.4,
            'per_beta_end': 1.0
        },
        'training': {
            'episodes': 10,
            'batch_size': 64,
            'save_frequency': 100
        },
        'logging': {
            'log_level': 'INFO',
            'tensorboard': True,
            'log_dir': 'logs/'
        }
    }


def save_default_config(output_path: Union[str, Path]) -> None:
    """Save default configuration to file.
    
    Args:
        output_path: Path to save default configuration
    """
    config = create_default_config()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)