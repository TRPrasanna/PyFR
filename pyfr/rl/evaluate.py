import torch
import torch.nn as nn
import numpy as np
from tensordict.nn import TensorDictModule
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator, NormalParamExtractor
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.envs import (
    Compose,
    StepCounter,
    TransformedEnv,
)
import matplotlib.pyplot as plt
from .train import HyperParameters
from pyfr.inifile import Inifile
from pyfr.readers.native import NativeReader
from pyfr.rl.env import PyFREnvironment

def evaluate_policy(mesh_file, cfg_file, backend_name, model_path, restart_soln=None, episodes=1):
    """Evaluate trained policy"""
    device = torch.device('cuda')

    cfg = Inifile.load(cfg_file)
    mesh = NativeReader(mesh_file)
    if 'neuralnetwork-hyperparameters' not in cfg.sections():
        print("No neuralnetwork-hyperparameters section found in config file. Proceeding to use default hyperparameters.")

    hp = HyperParameters.from_config(cfg)

    env = PyFREnvironment(mesh, cfg, backend_name, restart_soln)
    env = TransformedEnv(
        env,
        Compose(
            StepCounter(),
        ),
    )

    # Load policy
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    
    actor_net = nn.Sequential(
        nn.Linear(env.observation_spec["observation"].shape[0], 512),
        nn.Tanh(),
        nn.Linear(hp.num_cells_policy, hp.num_cells_policy),
        nn.Tanh(),
        nn.Linear(hp.num_cells_policy, 2),
        NormalParamExtractor()
    ).to(device)
    
    actor_module = TensorDictModule(
        actor_net,
        in_keys=["observation"],
        out_keys=["loc", "scale"]
    ).to(device)
    
    policy = ProbabilisticActor(
        module=actor_module,
        spec=env.action_spec,
        in_keys=["loc", "scale"],
        distribution_class=TanhNormal,
        return_log_prob=True,
        distribution_kwargs={
        "low": env.action_spec.space.low,
        "high": env.action_spec.space.high,
        },
        #safe=True
    ).to(device)
    
    policy.load_state_dict(checkpoint['policy_state_dict'])
    policy.eval()
    
    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        try:
            print("Starting evaluation...")
            eval_rollout = env.rollout(100000, policy) # rollout will be stopped by tend
            
            # Extract actions and rewards
            actions = eval_rollout["action"].cpu().numpy()
            rewards = eval_rollout["next", "reward"].cpu().numpy()
            times = eval_rollout["next", "step_count"].cpu().numpy()
            
            print("\nAction history:")
            print("Time\t\tAction\t\tReward")
            print("-" * 40)
            
            # Handle array formatting explicitly
            for t in range(len(actions)):
                action_val = float(actions[t].flatten()[0])  # Extract single value
                reward_val = float(rewards[t])
                print(f"{t*env.action_interval:.2f}\t\t{action_val:.3f}\t\t{reward_val:.3f}")
            
            eval_reward = float(np.mean(rewards))  # Convert to float
            print(f"\nMean reward: {eval_reward:.4f}")

            # Create time array
            time_array = np.arange(len(actions)) * env.action_interval
            
            # Create figure with 2 subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # Plot action vs time
            ax1.plot(time_array, actions.flatten(), 'b-', label='Action')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Action')
            ax1.grid(True)
            ax1.legend()
            
            # Plot reward vs time
            ax2.plot(time_array, rewards, 'r-', label='Reward')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Reward')
            ax2.grid(True)
            ax2.legend()
            
            # Adjust layout and save
            plt.tight_layout()
            plt.savefig('evaluation_results.png')
            plt.close()
            
            print(f"\nPlots saved as evaluation_results.png")

            return eval_reward
            
        except RuntimeError as e:
            print(f"Evaluation failed: {str(e)}")
            return None
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            return None