import torch
from torchrl.envs.common import EnvBase
from torchrl.data import Bounded, Composite, Unbounded, Categorical
from tensordict import TensorDict
from pyfr.backends import get_backend
from pyfr.rank_allocator import get_rank_allocation
from pyfr.solvers import get_solver
import numpy as np

class PyFREnvironment(EnvBase):
    """PyFR environment compatible with TorchRL."""
    
    def __init__(self, mesh, cfg, backend_name, restart_soln=None):
        #device = torch.device('cuda' if backend_name in ['cuda', 'hip'] else 'cpu')
        #device = torch.device('cpu')
        device = torch.device('cuda')
        super().__init__(device=device)

        # Load mesh and config once
        self.mesh = mesh #NativeReader(mesh_file)
        self.cfg = cfg
        self.backend = get_backend(backend_name, self.cfg)
        self.rallocs = get_rank_allocation(self.mesh, self.cfg)

        self.tend = self.cfg.getfloat('solver-time-integrator', 'tend')
        self.num_control_actions = self.cfg.getint('solver-plugin-reinforcementlearning', 'num_control_actions')
        self.actions_low = self.cfg.getliteral('solver-plugin-reinforcementlearning', 'actions_low')
        self.actions_high = self.cfg.getliteral('solver-plugin-reinforcementlearning', 'actions_high')
        # check if there are num_control_actions action_lows and action_highs
        assert len(self.actions_low) == self.num_control_actions
        assert len(self.actions_high) == self.num_control_actions

        print(f"Number of control actions: {self.num_control_actions}")
        for i in range(self.num_control_actions):
            print(f"Control action {i+1} range: {self.actions_low[i]} to {self.actions_high[i]}")

        # Add global control signals storage array; initialize with action space low
        self.current_control = np.array(self.actions_low)

        # Initialize the solver and other components
        self.restart_soln = restart_soln
        self._init_solver() # probably hard to get observation_size without doing this

        # Get observation size from RL plugin
        obs_size = self.rl_plugin.observation_size
        print(f"Observation size: {obs_size}")

        # *_specs
        self.observation_spec = Composite(
            {
                "observation": Unbounded(
                    shape=(obs_size,),
                    device=self.device
                )
            },
            shape=torch.Size([])
        )

        self.state_spec = self.observation_spec.clone() # not sure if this is correct

        self.action_spec = Composite(
            {"action": Bounded(
                low=torch.tensor(self.actions_low, device=self.device),
                high=torch.tensor(self.actions_high, device=self.device),
                shape=(self.num_control_actions,),
                device=self.device
            )},
            batch_size=torch.Size([])
        )

        self.reward_spec = Composite(
            {
                "reward": Unbounded(shape=(1,), device=self.device)
            },
            shape=torch.Size([])
        )

        # Done specifications
        self.full_done_spec = Composite(
            {
                # Generic done flag (legacy)
                "done": Categorical(
                    n=2, 
                    shape=(1,), 
                    dtype=torch.bool, 
                    device=self.device
                ),
                # Natural termination (crashes, divergence)
                "terminated": Categorical(
                    n=2, 
                    shape=(1,), 
                    dtype=torch.bool, 
                    device=self.device
                ),
                # Time limit reached
                "truncated": Categorical(
                    n=2, 
                    shape=(1,), 
                    dtype=torch.bool, 
                    device=self.device
                ),
            },
            shape=torch.Size([])
        )
        print("Environment initialized.")

    def _init_solver(self):
        self.solver = get_solver(self.backend, self.rallocs, self.mesh, 
                               initsoln=self.restart_soln, cfg=self.cfg, env=self)
        self.rl_plugin = next(p for p in self.solver.plugins 
                            if p.name == 'reinforcementlearning')
        self.action_interval = self.rl_plugin.action_interval
        self.current_time = self.solver.tcurr
        self.next_action_time = self.current_time + self.action_interval

    def _get_observation_size(self):
        # Implement logic to determine observation size
        return self.rl_plugin.observation_size

    # Mandatory methods: _step, _reset and _set_seed

    def _reset(self, tensordict=None, **kwargs):
        #print("Reset called")
        self._init_solver()
        #self.rl_plugin.reset()
        
        shape = torch.Size([])
        state = self.state_spec.zero(shape)
        return state.update(self.full_done_spec.zero(shape))

    def _step(self, tensordict):
        # Update global control signals
        self.current_control = tensordict["action"].cpu().numpy()
        print(f"Step called with actions: {tensordict['action'].cpu().numpy()}")

        self.current_time = self.solver.tcurr
        # Update the next action time
        self.next_action_time += self.action_interval

        try:
            self.solver.advance_to(self.next_action_time)
            crashed = False
            # Normal reward computation
            reward = self._compute_reward()
            observation = self._get_observation()
        except RuntimeError as e:
            print(f"Solver crashed: {str(e)}. Resetting. Last actions were: {self.current_control}")
            crashed = True
            # Return neutral state with zero reward
            observation = self.observation_spec.zero(torch.Size([]))["observation"]
            reward = -10.0

        done = crashed or self.current_time >= self.tend
        terminated = crashed  
        truncated = not crashed and self.current_time >= self.tend

        return TensorDict({
            'observation': observation,
            'reward': torch.tensor([reward], device=self.device),
            'done': torch.tensor([done], device=self.device, dtype=torch.bool),
            'terminated': torch.tensor([terminated], device=self.device, dtype=torch.bool),
            'truncated': torch.tensor([truncated], device=self.device, dtype=torch.bool),
        }, batch_size=torch.Size([]))

    def _get_observation(self):
        # Get raw observation
        obs = self.rl_plugin._get_observation(self.solver)
        #print(f"Raw observation: {obs}")
        # obs from PyFR may be 32 or 64 bit float, need to check
        return obs
    
    def _compute_reward(self):
        # Retrieve reward from RL plugin
        reward = self.rl_plugin._get_reward(self.solver)
        return reward

    def _check_done(self): # not used now
        # Implement your termination condition
        max_time = self.cfg.getfloat('solver-time-integrator', 'tend')
        return self.current_time >= max_time

    def _set_seed(self, seed):
        torch.manual_seed(seed)

    def close(self):
        """Clean up resources"""
        # Don't finalize MPI here since other parts might still need it
        #self.solver = None
        #self.rl_plugin = None