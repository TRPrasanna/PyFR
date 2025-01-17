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

def evaluate_policy(mesh_file, cfg_file, backend_name, load_model, ic_dir=None, episodes=1):
    """Evaluate trained policy"""
    device = torch.device('cuda')

    cfg = Inifile.load(cfg_file)
    mesh = NativeReader(mesh_file)
    if 'neuralnetwork-hyperparameters' not in cfg.sections():
        print("No neuralnetwork-hyperparameters section found in config file. Proceeding to use default hyperparameters.")

    hp = HyperParameters.from_config(cfg)

    env = PyFREnvironment(mesh, cfg, backend_name, ic_dir=ic_dir)
    env = TransformedEnv(env,StepCounter())

    # Load policy
    checkpoint = torch.load(load_model, map_location=device, weights_only=True)
    
    actor_net = nn.Sequential(
        nn.Linear(env.observation_spec["observation"].shape[0], 512),
        nn.Tanh(),
        #nn.ReLU(),
        nn.Linear(hp.num_cells_policy, hp.num_cells_policy),
        #nn.ReLU(),
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
        return_log_prob=False, # True for PPO
        distribution_kwargs={
        "low": env.action_spec.space.low,
        "high": env.action_spec.space.high,
        },
        #safe=True
    ).to(device)
    
    policy.load_state_dict(checkpoint['policy_state_dict'])
    policy.eval()

    # Get stored rewards
    current_reward = checkpoint.get('current_reward', checkpoint.get('reward', None))
    best_reward = checkpoint.get('best_reward', current_reward)
    saved_episode = checkpoint.get('episode', 0)
    best_episode = checkpoint.get('best_episode', saved_episode)

    print("\nModel Information:")
    print("-" * 40)
    if current_reward is not None:
        print(f"Current reward: {current_reward:.4f}")
    if best_reward is not None:
        print(f"Best reward: {best_reward:.4f}")
        print(f"Best reward at episode: {best_episode}")
    print(f"Model saved at episode: {saved_episode}")
    print(f"Model path: {load_model}")

    # Set evaluation mode and run
    env.set_evaluation_mode(True)
    try:
        with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
            print("Starting evaluation...")
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
            header = f"{'Time':>10}"
            for i in range(num_actions):
                header += f"{'Action_'+str(i):>15}"
            header += f"{'Reward':>15}"
            print(header)
            print("-" * (10 + 15 * (num_actions + 1)))
            
            # Format and print data rows
            for t in range(len(time_array)):
                row = f"{time_array[t]:10.2f}"
                for i in range(num_actions):
                    row += f"{actions[t,i]:15.4f}"
                row += f"{rewards[t]:15.4f}"
                print(row)
            
             # Print evaluation results
            eval_reward = float(np.mean(rewards))
            print("\nEvaluation Results:")
            print("-" * 40)
            print(f"Expected reward: {current_reward:.4f}")
            print(f"Actual reward:   {eval_reward:.4f}")
            if current_reward is not None:
                print(f"Difference:      {((eval_reward - current_reward)/current_reward)*100:.2f}%")
            
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
            plt.savefig('evaluation_results.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"\nPlots saved as evaluation_results.png")
            del eval_rollout
            return eval_reward
            
    except Exception as e:
        print(f"Unexpected error in evaluation: {str(e)}")
        raise
    finally:
        env.set_evaluation_mode(False)