import torch
import numpy as np
import os
import sys
import time
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.envs import (
    Compose,
    StepCounter,
    DoubleToFloat,
    TransformedEnv,
)
import matplotlib.pyplot as plt
from .train import HyperParameters, compare_configs, make_policy_and_value_modules
from pyfr.inifile import Inifile
from pyfr.readers.native import NativeReader
from pyfr.rl.env import PyFREnvironment


def evaluate_policy(mesh_file, cfg_file, backend_name, load_model, ic_dir=None, episodes=1):
    """Evaluate trained policy"""
    # Get config path at the start
    if hasattr(cfg_file, 'name'):
        cfg_path = cfg_file.name
    else:
        cfg_path = cfg_file

    # Read the config file content for comparison
    try:
        with open(cfg_path, 'r') as f:
            config_content = f.read()
    except Exception as e:
        print(f"Warning: Could not read config file: {e}")
        config_content = None

    # Initialize environment
    env = PyFREnvironment(mesh_file, cfg_path, backend_name, device_id=0, 
                          ic_dir=ic_dir, print_diagnostic=True)
    env = TransformedEnv(env, StepCounter())

    # Load model checkpoint
    if not os.path.exists(load_model):
        print(f"Error: Model file not found: {load_model}")
        sys.exit(1)
        
    device = torch.device('cpu')  # Use CPU for evaluation by default
    checkpoint = torch.load(load_model, map_location=device)
    
    # First try to use hyperparameters from checkpoint, fall back to config file
    if 'hyperparameters' in checkpoint:
        print("Using hyperparameters from checkpoint")
        hp_dict = checkpoint['hyperparameters']
        hp = HyperParameters()
        for key, value in hp_dict.items():
            if hasattr(hp, key):
                setattr(hp, key, value)
        # Still calculate derived parameters
        hp._calculate_derived(env)
    else:
        print("No hyperparameters in checkpoint, using values from config file")
        if 'neuralnetwork-hyperparameters' not in env.cfg.sections():
            print("No neuralnetwork-hyperparameters section found in config file. Using default hyperparameters.")
        hp = HyperParameters.from_config(env.cfg)
        hp._calculate_derived(env)

    # Compare config files if both are available
    if 'config_content' in checkpoint and config_content:
        print("\nVerifying config files...")
        config_differences = compare_configs(checkpoint['config_content'], config_content)
        
        if config_differences:
            print("\nWARNING: Config file differences detected between checkpoint and current:")
            for line_num, ckpt_line, curr_line in config_differences:
                print(f"Line {line_num}:")
                print(f"  Checkpoint: {ckpt_line}")
                print(f"  Current:    {curr_line}")
                print()
        else:
            print("Config files match between checkpoint and current settings.")

    # Print config if flag is set
    if hasattr(hp, 'print_config_on_load') and hp.print_config_on_load and 'config_content' in checkpoint:
        print("\n=== CHECKPOINT CONFIG FILE CONTENT ===\n")
        print(checkpoint['config_content'])
        print("\n=======================================\n")

    policy, _ = make_policy_and_value_modules(
        env, hp, device, return_log_prob=False
    )
    
    policy.load_state_dict(checkpoint['policy_state_dict'])
    policy.eval()  # Set to evaluation mode

    # Get stored rewards and episodes
    current_reward = checkpoint.get('current_reward', checkpoint.get('reward', None))
    best_reward = checkpoint.get('best_reward', current_reward)
    saved_episode = checkpoint.get('episode', 0)
    best_episode = checkpoint.get('best_episode', saved_episode)
    batch_idx = checkpoint.get('batch_idx', None)

    print("\nModel Information:")
    print("-" * 40)
    if current_reward is not None:
        print(f"Current reward: {current_reward:.4f}")
    if best_reward is not None:
        print(f"Best reward: {best_reward:.4f}")
        print(f"Best reward at episode: {best_episode}")
    print(f"Model saved at episode: {saved_episode}")
    if batch_idx is not None:
        print(f"Model saved at batch: {batch_idx}")
    print(f"Model path: {load_model}")

    # Print network architecture summary
    print("\nNetwork Architecture:")
    print("-" * 40)
    action_dim = env.action_spec_unbatched.shape[-1]
    input_shape = env.observation_spec["observation"].shape
    print(f"Input shape: {input_shape}")
    print(f"Output shape: {action_dim}")
    print(f"Policy architecture: {hp.policy_architecture}")
    print(f"Value architecture: {hp.value_architecture}")
    print(f"Hidden layers: {hp.num_hidden_layers_policy}")
    print(f"Hidden units: {hp.num_cells_policy}")
    print(f"Activation: {hp.activation_policy}")
    print(f"State-independent normal scale: {hp.state_ind_normal_scale}")

    # Set evaluation mode and run
    env.set_evaluation_mode(True)
    try:
        with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
            print("\nStarting evaluation...")
            eval_rollout = env.rollout(100000, policy)
            
            # Extract data and process for plotting
            actions = eval_rollout["action"].cpu().numpy()
            rewards = eval_rollout["next", "reward"].cpu().numpy().flatten()
            
            # Handle single vs multi-action case
            if len(actions.shape) == 1:
                actions = actions.reshape(-1, 1)
            num_actions = actions.shape[1]
            time_array = np.arange(len(actions)) * env.action_interval
            
            # Print action history
            print("\nAction history:")
            column_width = 16
            header = f"{'Time':>{column_width}}"
            for i in range(num_actions):
                header += f"{('Action_'+str(i)):>{column_width}}"
            header += f"{'Reward':>{column_width}}"
            print(header)
            print("-" * (column_width * (num_actions + 2)))
            
            # Format and print data rows
            for t in range(len(time_array)):
                row = f"{time_array[t]:>{column_width}.7e}"
                for i in range(num_actions):
                    row += f"{actions[t,i]:>{column_width}.7e}"
                row += f"{rewards[t]:>{column_width}.7e}"
                print(row)
            
            # Calculate statistics for rewards
            eval_reward = float(np.mean(rewards))
            eval_std = float(np.std(rewards))
            eval_min = float(np.min(rewards))
            eval_max = float(np.max(rewards))
            eval_total = float(np.sum(rewards))
            
            # Print evaluation results with more statistics
            print("\nEvaluation Results:")
            print("-" * 40)
            print(f"Expected reward: {current_reward:.4f}")
            print(f"Actual mean reward: {eval_reward:.4f}")
            print(f"Reward std dev: {eval_std:.4f}")
            print(f"Min/Max rewards: {eval_min:.4f} / {eval_max:.4f}")
            print(f"Total reward: {eval_total:.4f}")
            print(f"Number of steps: {len(rewards)}")
            
            if current_reward is not None:
                print(f"Difference from expected: {((eval_reward - current_reward)/current_reward)*100:.2f}%")
            
            # Create evaluation plots
            fig, axes = plt.subplots(num_actions + 1, 1, 
                                   figsize=(12, 4*(num_actions + 1)),
                                   sharex=True)
            axes = np.atleast_1d(axes)
            
            for i in range(num_actions):
                axes[i].plot(time_array, actions[:,i], '-', label=f'Action {i}')
                axes[i].set_ylabel(f'Action {i}')
                axes[i].grid(True)
                axes[i].legend()
            
            axes[-1].plot(time_array, rewards, 'r-', label='Reward')
            axes[-1].set_xlabel('Time')
            axes[-1].set_ylabel('Reward')
            axes[-1].grid(True)
            axes[-1].legend()

            plt.tight_layout()
            
            # Create output filename with timestamp
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            plot_filename = f'evaluation_results_{timestamp}.png'
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"\nPlots saved as {plot_filename}")
            del eval_rollout
            return eval_reward
            
    except Exception as e:
        print(f"Unexpected error in evaluation: {str(e)}")
        raise
    finally:
        env.set_evaluation_mode(False)
