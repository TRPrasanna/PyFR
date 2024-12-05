import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from tensordict.nn import TensorDictModule
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator, NormalParamExtractor
from torchrl.collectors import SyncDataCollector
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

def evaluate_policy(env, model_path, num_episodes=10):
    """Evaluate trained policy"""
    device = env.device
    
    # Load best model
    checkpoint = torch.load(model_path, map_location=device)
    
    # Recreate policy network
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
    
    # Load weights
    policy.load_state_dict(checkpoint['policy_state_dict'])
    policy.eval()
    
    # Evaluation loop
    rewards = []
    with torch.no_grad():
        for ep in tqdm(range(num_episodes), desc="Evaluating"):
            state = env.reset()
            episode_reward = 0
            
            done = False
            while not done:
                action = policy.act(state, deterministic=True)
                state = env.step(action)
                reward = state["reward"].item()
                episode_reward += reward
                done = state["done"].item()
                
            rewards.append(episode_reward)
            print(f"Episode {ep+1}: Reward = {episode_reward:.4f}")
    
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    print(f"\nEvaluation over {num_episodes} episodes:")
    print(f"Mean reward: {mean_reward:.4f} ± {std_reward:.4f}")
    
    return mean_reward, std_reward