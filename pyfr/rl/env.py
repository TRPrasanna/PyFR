import torch
from torchrl.envs.common import EnvBase
from torchrl.data import Bounded, Composite, Unbounded, Categorical
from tensordict import TensorDict, TensorDictBase
from pyfr.inifile import Inifile
from pyfr.backends import get_backend
from pyfr.readers.native import NativeReader
from pyfr.rank_allocator import get_rank_allocation
from pyfr.solvers import get_solver
from typing import Optional

class PyFREnvironment(EnvBase):
    """PyFR environment compatible with TorchRL."""
    
    def __init__(self, mesh_file, cfg_file, backend_name, restart_soln=None):
        #device = torch.device('cuda' if backend_name in ['cuda', 'hip'] else 'cpu')
        device = torch.device('cpu')
        super().__init__(device=device)

        # Load mesh and config once
        self.mesh = NativeReader(mesh_file)
        self.cfg = Inifile.load(cfg_file)
        #print(self.cfg)
        self.backend = get_backend(backend_name, self.cfg)
        self.rallocs = get_rank_allocation(self.mesh, self.cfg)

        # Initialize the solver and other components
        self.restart_soln = restart_soln
        self._init_solver()

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
                low=torch.tensor(-0.09, device=self.device), # not sure if this is the best way
                high=torch.tensor(0.09, device=self.device),
                shape=(1,),
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

        self.full_done_spec = Composite(
            {
                "done": Categorical(n=2, shape=(1,), dtype=torch.bool, device=self.device),
            },
            shape=torch.Size([])
        )
        self.full_done_spec["terminated"] = self.full_done_spec["done"].clone()
        self.full_done_spec["truncated"] = self.full_done_spec["done"].clone()
        print("Environment initialized.")

    def _init_solver(self):
        self.solver = get_solver(self.backend, self.rallocs, self.mesh, 
                               initsoln=self.restart_soln, cfg=self.cfg)
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
        print("Reset called")
        self._init_solver()
        #self.rl_plugin.reset()
        
        shape = torch.Size([])
        state = self.state_spec.zero(shape)
        return state.update(self.full_done_spec.zero(shape))

    def _step(self, tensordict):
        action = tensordict['action']
        #print(f"Step called with action: {action.item()}")
        
        # Set the action in the RL plugin
        self.rl_plugin.set_action(action.item())
        # Advance the solver to the next action time
        #t_end = self.next_action_time
        #while self.solver.tcurr < t_end:
        #    self.solver.advance_to(self.solver.tcurr + self.solver._dt)
        #    self.current_time = self.solver.tcurr
        self.solver.advance_to(self.next_action_time)
        self.current_time = self.solver.tcurr
        # Update the next action time
        self.next_action_time += self.action_interval
        # Get observation and reward
        observation = self._get_observation()
        reward = self._compute_reward()
        done = self._check_done()

        #print observation
        #print(f"Observation: {observation}")
        return TensorDict({
            'observation': observation,
            'reward': torch.tensor([reward], device=self.device),
            'done': torch.tensor([done], device=self.device, dtype=torch.bool),
            'terminated': torch.tensor([done], device=self.device, dtype=torch.bool),
            'truncated': torch.tensor([False], device=self.device, dtype=torch.bool),
        }, batch_size=torch.Size([]))

    def _get_observation(self):
        # Get raw observation
        obs_np = self.rl_plugin._get_observation(self.solver)
        
        # Proper tensor construction with dtype
        obs = torch.as_tensor(obs_np, 
                            device=self.device, 
                            dtype=torch.float32) # obs from PyFR may be 32 or 64 bit float
        return obs
    
    def _compute_reward(self):
        # Retrieve reward from RL plugin
        reward = self.rl_plugin._get_reward(self.solver)
        return reward

    def _check_done(self):
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