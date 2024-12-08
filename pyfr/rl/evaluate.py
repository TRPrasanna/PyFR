import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from tensordict.nn import TensorDictModule
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator, NormalParamExtractor
from torchrl.collectors import SyncDataCollector
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

def evaluate_policy(env, model_path, num_episodes=1):
    """Evaluate trained policy for same duration as training episode"""
    device = env.device
    
    # Get timing parameters from config
    tend = env.cfg.getfloat('solver-time-integrator', 'tend')
    tstart = env.cfg.getfloat('solver-time-integrator', 'tstart', 0.0)
    simulation_time = tend - tstart
    
    # Load and setup policy
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    
    actor_net = nn.Sequential(
        nn.Linear(env.observation_spec["observation"].shape[0], 512),
        nn.Tanh(),
        nn.Linear(512, 512),
        nn.Tanh(),
        nn.Linear(512, 2),
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
        safe=True
    ).to(device)
    
    policy.load_state_dict(checkpoint['policy_state_dict'])
    policy.eval()
    
    with torch.no_grad():
        try:
            print(f"Starting evaluation for {simulation_time} time units...")
            state = env.reset()
            total_reward = 0
            step = 0
            
            while env.current_time < tend:
                policy_output = policy(state)
                next_state = env.step(policy_output)
                reward = next_state["next", "reward"].item()
                total_reward += reward
                
                if step % 10 == 0:
                    action = policy_output["action"].item()
                    print(f"Time {env.current_time:.3f}: Action = {action:.3f}, Reward = {reward:.3f}")
                    #print(next_state["next", "observation"])
                
                state = next_state
                step += 1
            
            print(f"\nSimulation completed:")
            print(f"End time: {env.current_time:.3f}")
            print(f"Total steps: {step}")
            print(f"Final reward: {total_reward:.4f}")
            return total_reward, None
            
        except RuntimeError as e:
            print(f"Simulation failed: {str(e)}")
            return None, None