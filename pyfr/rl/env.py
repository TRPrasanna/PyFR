import torch
from torchrl.envs.common import EnvBase
from torchrl.data import Bounded, Composite, Unbounded, Categorical
from tensordict import TensorDict
from pyfr.backends import get_backend
from pyfr.rank_allocator import get_rank_allocation
from pyfr.solvers import get_solver
import numpy as np
import os
import random
from typing import List, Set
from pyfr.readers.native import NativeReader
from datetime import datetime

class PyFREnvironment(EnvBase):
    """PyFR environment compatible with TorchRL."""
    
    def __init__(self, mesh, cfg, backend_name, ic_dir=None):
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
        self.num_control_actions = self.cfg.getint('solver-plugin-reinforcementlearning', 'num-control-actions')
        self.actions_low = self.cfg.getliteral('solver-plugin-reinforcementlearning', 'actions-low')
        self.actions_high = self.cfg.getliteral('solver-plugin-reinforcementlearning', 'actions-high')
        # check if there are num_control_actions action_lows and action_highs
        assert len(self.actions_low) == self.num_control_actions
        assert len(self.actions_high) == self.num_control_actions

        print(f"Number of control actions: {self.num_control_actions}")
        for i in range(self.num_control_actions):
            print(f"Control action {i+1} range: {self.actions_low[i]} to {self.actions_high[i]}")

        # Add global control signals storage array; initialize with action space low
        self.current_control = np.array(self.actions_low)

        self.dtend = self.cfg.getfloat('solver-time-integrator', 'dtend') # difference between initial and final time
        # Get evaluation time if specified, otherwise use training time
        self.eval_time = self.cfg.getfloat('solver-plugin-reinforcementlearning', 
                                          'eval-time', 
                                          self.dtend)
        # Track current episode time limit
        self._current_time_limit = self.dtend

        self.action_interval = self.cfg.getfloat('solver-plugin-reinforcementlearning', 'action-interval')
        # Calculate max steps from time limits
        self.max_training_steps = int(self.dtend / self.action_interval)
        self.max_eval_steps = int(self.eval_time / self.action_interval)
        self._current_max_steps = self.max_training_steps
        self.step_count = -1 # matching this with inters.py to prevent if condition from being true

        # Initialize IC manager if directory provided
        self.ic_manager = None
        if ic_dir is not None:
            try:
                self.ic_manager = InitialConditionManager(ic_dir, mesh['mesh_uuid'])
            except ValueError as e:
                print(f"\nWarning: {str(e)}")
                print("Continuing without initial condition snapshots...")
        else:
            print("\nNote: No initial condition directory provided.")
            print("Training will use default initial conditions.")

        self.eval_ic = None  # Store evaluation IC
        self.is_evaluating = False  # Track evaluation mode

        # Initialize the solver and other components; no need to pass initial solution for now
        self.restart_soln = None 
        self._init_solver() # probably hard to get observation_size without doing this, but should preferably get rid of this

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

    def _init_solver(self, initsoln=None):
        self.restart_soln = initsoln
        self.solver = get_solver(self.backend, self.rallocs, self.mesh, 
                               initsoln=self.restart_soln, cfg=self.cfg, env=self)
        self.rl_plugin = next(p for p in self.solver.plugins 
                            if p.name == 'reinforcementlearning')      
        self.current_time = self.solver.tcurr
        # Use current mode's time limit
        self.max_time = self.current_time + self._current_time_limit

    def _get_observation_size(self):
        # Implement logic to determine observation size
        return self.rl_plugin.observation_size

    # Mandatory methods: _step, _reset and _set_seed

    def _reset(self, tensordict=None, **kwargs):
        print("Reset called")
        self.step_count = 0

        restart_soln = None
        # Handle evaluation mode differently
        if self.ic_manager is not None:
            try:
                if self.is_evaluating:
                    restart_soln = self.ic_manager.get_eval_ic()
                else:
                    ic_file = self.ic_manager.get_random_ic()
                    restart_soln = NativeReader(ic_file)
            except Exception as e:
                print(f"Warning: Failed to load IC file: {str(e)}")
                print("Using default initial conditions.")

        self._init_solver(initsoln=restart_soln)
        #self.rl_plugin.reset()
        
        shape = torch.Size([])
        state = self.state_spec.zero(shape)
        return state.update(self.full_done_spec.zero(shape))

    def _step(self, tensordict):
        # Update global control signals
        self.current_control = tensordict["action"].cpu().numpy()
        self.step_count = tensordict["step_count"].item()
        
        self.current_time = self.solver.tcurr
        #print(f"Step called with actions: {tensordict['action'].cpu().numpy()} at step {self.step_count} and time {self.current_time}")
        # Update the next action time
        self.next_action_time = self.current_time + self.action_interval
        #print(f"Stepcount: {self.step_count}, Current time: {self.current_time}, going to advance to {self.next_action_time}")

        try:
            self.solver.advance_to(self.next_action_time)
            #print(f"Advanced to {self.solver.tcurr}")
            crashed = False
            # Normal reward computation
            reward = self._compute_reward()
            observation = self._get_observation()
            truncated = self._check_done()
            terminated = False

        except RuntimeError as e:
            print(f"Solver crashed: {str(e)}. Resetting. Last actions were: {self.current_control}")
            crashed = True
            # Return neutral state with zero reward
            observation = self.observation_spec.zero(torch.Size([]))["observation"]
            reward = -10.0
            truncated = False
            terminated = True

        return TensorDict({
            'observation': observation,
            'reward': torch.tensor([reward], device=self.device),
            'done': torch.tensor([terminated or truncated], device=self.device, dtype=torch.bool),
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

    def _check_done(self) -> bool:
        """Check if episode should terminate based on step count."""
        # this sets the done state but resets are also handled by frames_per_batch in collector
        #print(f"Checking done at the end of step {self.step_count}")
        #if(self.step_count +1 >= self._current_max_steps):
        #    print("We are done")
        #else:
        #    print("We are not done")
        return self.step_count + 1 >= self._current_max_steps

    def _set_seed(self, seed): # this does nothing for now
        torch.manual_seed(seed)

    def set_evaluation_mode(self, is_evaluating: bool):
        """Set environment to evaluation mode."""
        self.is_evaluating = is_evaluating
        self._current_time_limit = self.eval_time if is_evaluating else self.dtend
        self._current_max_steps = self.max_eval_steps if is_evaluating else self.max_training_steps

    def close(self):
        """Clean up resources"""
        # Don't finalize MPI here since other parts might still need it
        #self.solver = None
        #self.rl_plugin = None

class InitialConditionManager:
    def __init__(self, ic_dir: str, mesh_uuid: str):
        self.ic_dir = ic_dir
        self.mesh_uuid = mesh_uuid
        self.ic_files = self._find_valid_ics()
        self.unused_files = set(self.ic_files)
        self.eval_ic_file = self._get_oldest_ic()
        print(f"\nFound {len(self.ic_files)} initial condition files in {ic_dir}")
        if self.eval_ic_file:
            print(f"Using {os.path.basename(self.eval_ic_file)} for evaluation")

    def _get_oldest_ic(self) -> str:
        """Get oldest .pyfrs file by creation time."""
        if not self.ic_files:
            return None
        oldest = min(self.ic_files, key=os.path.getctime)
        creation_time = datetime.fromtimestamp(os.path.getctime(oldest))
        print(f"\nSelected evaluation IC: {os.path.basename(oldest)}")
        print(f"Creation time: {creation_time:%Y-%m-%d %H:%M:%S}")
        return oldest

    def get_eval_ic(self) -> NativeReader:
        """Get the evaluation IC."""
        return NativeReader(self.eval_ic_file) if self.eval_ic_file else None
        
    def _find_valid_ics(self) -> List[str]:
        """Find all valid .pyfrs files in directory."""
        if not os.path.exists(self.ic_dir):
            raise ValueError(f"IC directory {self.ic_dir} not found")
            
        ic_files = []
        for f in os.listdir(self.ic_dir):
            if f.endswith('.pyfrs'):
                file_path = os.path.join(self.ic_dir, f)
                try:
                    soln = NativeReader(file_path)
                    if soln['mesh_uuid'] == self.mesh_uuid:
                        ic_files.append(file_path)
                except:
                    continue
        return ic_files

    def get_random_ic(self) -> str:
        """Get random unused IC file, reset if all used."""
        if not self.unused_files:
            self.unused_files = set(self.ic_files)
        
        ic_file = random.choice(list(self.unused_files))
        self.unused_files.remove(ic_file)
        return ic_file