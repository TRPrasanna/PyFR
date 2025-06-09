import torch
import torch.nn as nn
import numpy as np
import os
import time
import functools
import math
from tensordict.nn import TensorDictModule, AddStateIndependentNormalScale, TensorDictSequential
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator, NormalParamExtractor, LSTMModule, MLP, set_recurrent_mode
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.envs import (
    Compose,
    StepCounter,
    DoubleToFloat,
    TransformedEnv,
    InitTracker,
)
import matplotlib.pyplot as plt
from .train import HyperParameters
from pyfr.inifile import Inifile
from pyfr.readers.native import NativeReader
from pyfr.rl.env import PyFREnvironment

def evaluate_policy(mesh_file, cfg_file, backend_name, load_model, ic_dir=None, episodes=1):
    """Evaluate trained LSTM-enabled PPO policy"""
    #device = torch.device('cuda')
    device = torch.device('cpu')

    # Get config path at the start
    if hasattr(cfg_file, 'name'):
        cfg_path = cfg_file.name
    else:
        cfg_path = cfg_file

    env = PyFREnvironment(mesh_file, cfg_path, backend_name, device_id=0, ic_dir=ic_dir, print_diagnostic=True)
    # Add InitTracker for LSTM recurrent states
    env = TransformedEnv(env, Compose(StepCounter(), InitTracker()))

    if 'neuralnetwork-hyperparameters' not in env.cfg.sections():
        print("No neuralnetwork-hyperparameters section found in config file. Proceeding to use default hyperparameters.")

    hp = HyperParameters.from_config(env.cfg)

    # Load policy
    checkpoint = torch.load(load_model, map_location=device, weights_only=True)
    
    # Actor network with LSTM - MUST match train.py exactly
    action_dim = env.action_spec_unbatched.shape[-1]
    input_shape = env.observation_spec["observation"].shape
    
    # LSTM hidden size
    lstm_hidden_size = hp.num_cells_policy
    
    # Actor network with LSTM - using simpler pattern (matching train.py)
    # Step 1: Preprocessing MLP to transform observation for LSTM
    actor_preprocessing = TensorDictModule(
        MLP(
            in_features=input_shape[-1],
            out_features=lstm_hidden_size,
            num_cells=[lstm_hidden_size],
            activation_class=nn.Tanh,
            device=device,
        ),
        in_keys=["observation"],
        out_keys=["_actor_embed"]
    )
    
    # Step 2: LSTM with simple in/out keys
    actor_lstm = LSTMModule(
        input_size=lstm_hidden_size,
        hidden_size=lstm_hidden_size,
        device=device,
        in_key="_actor_embed",
        out_key="_actor_embed",
        python_based=True,
    )
    
    # Step 3: Final MLP that sees both LSTM output and original observation
    actor_head = MLP(
        in_features=lstm_hidden_size + input_shape[-1],  # LSTM output + observation
        out_features=2 * action_dim,  # mean and scale parameters
        num_cells=[hp.num_cells_policy, hp.num_cells_policy],
        activation_class=nn.Tanh,
        device=device,
    )
    
    # Initialize actor head weights
    for layer in actor_head.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.orthogonal_(layer.weight, 1.0)
            layer.bias.data.zero_()
    
    # Create actor head module that outputs raw parameters
    actor_head_module = TensorDictModule(
        actor_head,
        in_keys=["_actor_embed", "observation"],  # Both LSTM output and observation
        out_keys=["params"]  # Raw parameters before extraction
    )
    
    # NormalParamExtractor to split parameters into loc and scale
    param_extractor = TensorDictModule(
        NormalParamExtractor(
            scale_mapping="biased_softplus_1.0",
            scale_lb=0.1,   # lower bound for scale
        ),
        in_keys=["params"],
        out_keys=["loc", "scale"]
    )
    
    # Create the full actor network using the simpler pattern
    actor_net = TensorDictSequential(
        actor_preprocessing,
        actor_lstm,
        actor_head_module,
        param_extractor
    ).to(device)

    policy = ProbabilisticActor(
        module=actor_net,
        spec=env.action_spec,
        in_keys=["loc", "scale"],
        distribution_class=TanhNormal,
        return_log_prob=True,
        distribution_kwargs={
            "low": env.action_spec.space.low,
            "high": env.action_spec.space.high,
            "tanh_loc": False,
        },
    ).to(device)

    # Value network (critic) with LSTM - using simpler pattern (matching train.py)
    # Step 1: Preprocessing MLP to transform observation for LSTM
    critic_preprocessing = TensorDictModule(
        MLP(
            in_features=input_shape[-1],
            out_features=hp.num_cells_value,
            num_cells=[hp.num_cells_value],
            activation_class=nn.Tanh,
            device=device,
        ),
        in_keys=["observation"],
        out_keys=["_critic_embed"]
    )
    
    # Step 2: LSTM with simple in/out keys
    critic_lstm = LSTMModule(
        input_size=hp.num_cells_value,
        hidden_size=hp.num_cells_value,
        device=device,
        in_key="_critic_embed",
        out_key="_critic_embed",
    )
    
    # Step 3: Final MLP that sees both LSTM output and original observation
    critic_head = MLP(
        in_features=hp.num_cells_value + input_shape[-1],  # LSTM output + observation
        out_features=1,
        num_cells=[hp.num_cells_value, hp.num_cells_value],
        activation_class=nn.Tanh,
        device=device,
    )
    
    critic_head_module = TensorDictModule(
        critic_head,
        in_keys=["_critic_embed", "observation"],  # Both LSTM output and observation
        out_keys=["state_value"]
    )
    
    value_net = TensorDictSequential(
        critic_preprocessing,
        critic_lstm,
        critic_head_module
    ).to(device)

    # Use value_net directly (no ValueOperator wrapper needed with simplified approach)

    # Load model weights
    policy.load_state_dict(checkpoint['policy_state_dict'])
    if 'value_state_dict' in checkpoint:
        value_net.load_state_dict(checkpoint['value_state_dict'])

    print(f"Loaded model from: {load_model}")
    if 'best_reward' in checkpoint:
        print(f"Model best reward: {checkpoint['best_reward']:.4f}")
    if 'episode' in checkpoint:
        print(f"Model episode: {checkpoint['episode']}")

    # Set environment step count to evaluation mode and policy to evaluation
    policy.eval()
    if value_net is not None:
        value_net.eval()

    # Run evaluation episodes
    episode_rewards = []
    episode_lengths = []
    all_actions = []
    all_observations = []
    
    print(f"\nRunning {episodes} evaluation episodes...")
    
    for episode in range(episodes):
        print(f"\nEpisode {episode + 1}/{episodes}")
        
        # Reset environment and ensure LSTM states are initialized
        with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC), set_recurrent_mode(False):
            td = env.reset()
            episode_reward = 0
            episode_length = 0
            episode_actions = []
            episode_observations = []
            
            # Run episode
            while not td["done"].any():
                # Store observation
                episode_observations.append(td["observation"].cpu().numpy())
                
                # Get action from policy
                td = policy(td)
                episode_actions.append(td["action"].cpu().numpy())
                
                # Take step
                td = env.step(td)
                
                # Accumulate reward
                reward = td["next", "reward"].item()
                episode_reward += reward
                episode_length += 1
                
                if episode_length >= 10000:  # Safety break
                    print("Episode length exceeded 10000 steps, breaking...")
                    break
                
                # Move to next state
                td = env.step_mdp(td)
            
            # Store episode results
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            all_actions.append(np.array(episode_actions))
            all_observations.append(np.array(episode_observations))
            
            print(f"Episode {episode + 1} - Reward: {episode_reward:.4f}, Length: {episode_length}")

    # Calculate statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    std_length = np.std(episode_lengths)

    print(f"\n{'='*60}")
    print("EVALUATION RESULTS (PPO-LSTM)")
    print(f"{'='*60}")
    print(f"Episodes: {episodes}")
    print(f"Mean Reward: {mean_reward:.4f} ± {std_reward:.4f}")
    print(f"Mean Episode Length: {mean_length:.2f} ± {std_length:.2f}")
    print(f"Min Reward: {min(episode_rewards):.4f}")
    print(f"Max Reward: {max(episode_rewards):.4f}")
    print(f"{'='*60}")

    # Return results
    results = {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'all_actions': all_actions,
        'all_observations': all_observations,
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'mean_length': mean_length,
        'std_length': std_length,
    }

    return results


def plot_evaluation_results(results, save_path=None):
    """Plot evaluation results"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Episode rewards
    axes[0, 0].plot(results['episode_rewards'], 'b-', linewidth=2)
    axes[0, 0].axhline(y=results['mean_reward'], color='r', linestyle='--', 
                       label=f'Mean: {results["mean_reward"]:.2f}')
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Episode lengths
    axes[0, 1].plot(results['episode_lengths'], 'g-', linewidth=2)
    axes[0, 1].axhline(y=results['mean_length'], color='r', linestyle='--',
                       label=f'Mean: {results["mean_length"]:.2f}')
    axes[0, 1].set_title('Episode Lengths')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Length')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Reward distribution
    axes[1, 0].hist(results['episode_rewards'], bins=10, alpha=0.7, color='blue')
    axes[1, 0].axvline(x=results['mean_reward'], color='r', linestyle='--',
                       label=f'Mean: {results["mean_reward"]:.2f}')
    axes[1, 0].set_title('Reward Distribution')
    axes[1, 0].set_xlabel('Reward')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Action analysis (if multi-dimensional)
    if len(results['all_actions']) > 0 and len(results['all_actions'][0]) > 0:
        actions_concat = np.concatenate(results['all_actions'], axis=0)
        if actions_concat.shape[1] > 1:
            # Multi-dimensional actions
            for i in range(min(3, actions_concat.shape[1])):  # Plot first 3 action dimensions
                axes[1, 1].plot(actions_concat[:, i], label=f'Action {i+1}', alpha=0.7)
            axes[1, 1].set_title('Action Trajectories')
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('Action Value')
            axes[1, 1].legend()
        else:
            # Single action dimension
            axes[1, 1].plot(actions_concat[:, 0], 'purple', alpha=0.7)
            axes[1, 1].set_title('Action Trajectory')
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('Action Value')
        axes[1, 1].grid(True)
    else:
        axes[1, 1].text(0.5, 0.5, 'No Action Data', ha='center', va='center',
                        transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Actions')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Evaluation plots saved to: {save_path}")
    
    plt.show()
    
    return fig
