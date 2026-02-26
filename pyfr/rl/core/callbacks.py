from dataclasses import asdict
import os
import sys
import time
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy as sb3_evaluate_policy
from tqdm.auto import tqdm

from .utils import save_metadata


def _evaluate_mean_step_reward(model, eval_env, n_eval_episodes: int,
                               recurrent: bool) -> float:
    if not recurrent:
        ep_rewards, ep_lengths = sb3_evaluate_policy(
            model,
            eval_env,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            render=False,
            return_episode_rewards=True,
            warn=False
        )
        total_reward = float(np.sum(ep_rewards))
        total_steps = int(np.sum(ep_lengths))
        return total_reward / total_steps if total_steps > 0 else 0.0

    total_reward = 0.0
    total_steps = 0
    n_envs = getattr(eval_env, 'num_envs', 1)

    for _ in range(n_eval_episodes):
        obs = eval_env.reset()
        lstm_states = None
        episode_starts = np.ones((n_envs,), dtype=bool)

        done = np.zeros((n_envs,), dtype=bool)
        while not done[0]:
            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=True
            )
            obs, rewards, done, _infos = eval_env.step(action)
            total_reward += float(rewards[0])
            total_steps += 1
            episode_starts = done

    return total_reward / total_steps if total_steps > 0 else 0.0


class SB3EvalAndCheckpointCallback(BaseCallback):
    def __init__(self, eval_env, hp, checkpoint_dir: str,
                 cfg_path: str, config_content: str | None,
                 algorithm_name: str = 'ppo',
                 recurrent_policy: bool = False,
                 start_episode: int = 0,
                 start_timesteps: int = 0,
                 best_eval_reward: float = float('-inf'),
                 verbose: int = 1):
        super().__init__(verbose=verbose)

        self.eval_env = eval_env
        self.hp = hp
        self.checkpoint_dir = checkpoint_dir
        self.cfg_path = cfg_path
        self.config_content = config_content
        self.algorithm_name = algorithm_name
        self.recurrent_policy = recurrent_policy
        self.start_episode = start_episode
        self.start_timesteps = start_timesteps

        self.best_eval_reward = best_eval_reward
        self.latest_eval_reward = None
        self.best_eval_episode = start_episode

        self.eval_freq_steps = max(1, hp.eval_frequency * hp.rollout_size)
        self._last_eval_timestep = 0

        self.best_model_path = os.path.join(checkpoint_dir, 'best-model.zip')
        self.latest_model_path = os.path.join(checkpoint_dir, 'latest-model.zip')

        # TorchRL-like progress reporting state
        self._pbar = None
        self._episodes_shown = max(0, int(start_episode))
        self._latest_train_reward = None
        self._latest_eval_str = ''
        self._is_tty = sys.stderr.isatty()
        self._training_start_walltime = None

    def _current_lr(self) -> float:
        try:
            return float(self.model.policy.optimizer.param_groups[0]['lr'])
        except Exception:
            return 0.0

    def _episodes_done(self) -> int:
        return int(self._total_timesteps() / max(1, self.hp.actions_per_episode))

    def _set_progress_postfix(self):
        if self._pbar is None or not self._is_tty:
            return

        postfix = {}
        if self._latest_train_reward is not None:
            postfix['train_reward'] = f'{self._latest_train_reward:.5f}'
        if self._latest_eval_str:
            postfix['eval'] = self._latest_eval_str
        postfix['lr'] = f'{self._current_lr():.2e}'

        self._pbar.set_postfix(postfix)

    def _sync_progress_bar(self):
        if self._pbar is None or not self._is_tty:
            return

        done = min(self.hp.episodes, self._episodes_done())
        if done > self._episodes_shown:
            self._pbar.update(done - self._episodes_shown)
            self._episodes_shown = done

    def _total_timesteps(self) -> int:
        # When resuming, some SB3 saves restore num_timesteps and some flows
        # restart it from zero. Make metadata monotonic in both cases.
        cur = int(self.model.num_timesteps)
        if cur < self.start_timesteps:
            return cur + self.start_timesteps
        return cur

    def _serializable_hparams(self) -> dict[str, Any]:
        hp_dict = asdict(self.hp)
        return {k: v for k, v in hp_dict.items() if not k.startswith('_')}

    def _save_with_metadata(self, model_path: str, eval_reward: float,
                            batch_idx: int, episode_idx: int):
        self.model.save(model_path)

        metadata = {
            'algorithm': self.algorithm_name,
            'current_reward': float(eval_reward),
            'best_reward': float(self.best_eval_reward),
            'episode': int(episode_idx),
            'best_episode': int(self.best_eval_episode),
            'batch_idx': int(batch_idx),
            'timesteps': int(self._total_timesteps()),
            'hyperparameters': self._serializable_hparams(),
            'config_content': self.config_content,
            'config_path': self.cfg_path,
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        save_metadata(model_path, metadata)

    def _run_eval_and_checkpoint(self):
        # TorchRL parity using native SB3 API:
        # mean per-step reward = sum(episode rewards) / sum(episode lengths)
        # over deterministic evaluation episodes.
        self.latest_eval_reward = _evaluate_mean_step_reward(
            self.model,
            self.eval_env,
            n_eval_episodes=1,
            recurrent=self.recurrent_policy
        )

        total_timesteps = self._total_timesteps()
        episodes_done = int(total_timesteps / max(1, self.hp.actions_per_episode))
        batch_idx = int(total_timesteps / max(1, self.hp.rollout_size))

        # Keep training output concise (TorchRL-like): do not print a
        # per-evaluation summary line here.

        model_batch_path = os.path.join(self.checkpoint_dir,
                                        f'model-{batch_idx}.zip')
        self._save_with_metadata(
            model_batch_path,
            self.latest_eval_reward,
            batch_idx,
            episodes_done
        )
        self._save_with_metadata(
            self.latest_model_path,
            self.latest_eval_reward,
            batch_idx,
            episodes_done
        )

        if self.latest_eval_reward > self.best_eval_reward:
            self.best_eval_reward = self.latest_eval_reward
            self.best_eval_episode = episodes_done
            if self.verbose:
                print(
                    f'New best eval reward: {self.best_eval_reward:.6f} '
                    f'at episode {episodes_done}'
                )
            self._save_with_metadata(
                self.best_model_path,
                self.latest_eval_reward,
                batch_idx,
                episodes_done
            )

        self._latest_eval_str = (
            f'eval reward: {self.latest_eval_reward:.5f} '
            f'(best: {self.best_eval_reward:.5f})'
        )
        self._set_progress_postfix()

    def _on_training_start(self):
        self._training_start_walltime = time.time()
        initial = min(max(0, self._episodes_shown), self.hp.episodes)
        self._episodes_shown = initial
        if self._is_tty:
            self._pbar = tqdm(total=self.hp.episodes, desc='Training', initial=initial)
            self._set_progress_postfix()
        else:
            self._pbar = None
        return None

    def _on_step(self):
        if self.num_timesteps - self._last_eval_timestep >= self.eval_freq_steps:
            self._run_eval_and_checkpoint()
            self._last_eval_timestep = int(self.num_timesteps)

        return True

    def _on_rollout_end(self):
        rb = getattr(self.model, 'rollout_buffer', None)
        if rb is not None and hasattr(rb, 'rewards'):
            try:
                self._latest_train_reward = float(rb.rewards.mean())
            except Exception:
                pass

        self._sync_progress_bar()
        self._set_progress_postfix()

        if not self._is_tty:
            episodes_done = min(self.hp.episodes, self._episodes_done())
            elapsed = (
                int(time.time() - self._training_start_walltime)
                if self._training_start_walltime is not None else 0
            )
            hh = elapsed // 3600
            mm = (elapsed % 3600) // 60
            ss = elapsed % 60

            train_str = (
                f'{self._latest_train_reward:.5f}'
                if self._latest_train_reward is not None else 'n/a'
            )
            eval_str = (
                f'{self.latest_eval_reward:.5f}'
                if self.latest_eval_reward is not None else 'n/a'
            )
            best_str = (
                f'{self.best_eval_reward:.5f}'
                if self.best_eval_reward != float('-inf') else 'n/a'
            )
            print(
                '[train] '
                f'episodes={episodes_done}/{self.hp.episodes} '
                f'train_reward={train_str} '
                f'eval={eval_str} '
                f'best={best_str} '
                f'lr={self._current_lr():.2e} '
                f'elapsed={hh:02d}:{mm:02d}:{ss:02d}'
            )
        return None

    def _on_training_end(self):
        # Ensure at least one checkpoint exists even if eval frequency is large.
        if self.latest_eval_reward is None:
            self._run_eval_and_checkpoint()

        self._sync_progress_bar()
        if self._pbar is not None and self._is_tty:
            self._pbar.close()


class SB3OptunaPruningCallback(BaseCallback):
    def __init__(self, trial, eval_env, hp,
                 recurrent_policy: bool = False,
                 n_eval_episodes: int = 1,
                 best_model_path: str | None = None,
                 verbose: int = 1):
        super().__init__(verbose=verbose)

        self.trial = trial
        self.eval_env = eval_env
        self.hp = hp
        self.recurrent_policy = recurrent_policy
        self.n_eval_episodes = max(1, int(n_eval_episodes))
        self.best_model_path = best_model_path

        self.eval_freq_steps = max(1, hp.eval_frequency * hp.rollout_size)
        self._last_eval_timestep = 0

        self.latest_eval_reward = None
        self.best_eval_reward = float('-inf')
        self.pruned = False
        self.pruned_update = None

    def _on_step(self):
        if self.num_timesteps - self._last_eval_timestep < self.eval_freq_steps:
            return True

        self.latest_eval_reward = _evaluate_mean_step_reward(
            self.model,
            self.eval_env,
            n_eval_episodes=self.n_eval_episodes,
            recurrent=self.recurrent_policy
        )

        update_idx = max(1, int(self.model.num_timesteps / max(1, self.hp.rollout_size)))
        self.trial.report(float(self.latest_eval_reward), step=update_idx)

        if self.latest_eval_reward > self.best_eval_reward:
            self.best_eval_reward = float(self.latest_eval_reward)
            if self.best_model_path is not None:
                self.model.save(self.best_model_path)

        if self.verbose:
            print(
                '[hpo] '
                f'trial={self.trial.number} '
                f'update={update_idx} '
                f'eval={self.latest_eval_reward:.5f} '
                f'best={self.best_eval_reward:.5f}'
            )

        if self.trial.should_prune():
            self.pruned = True
            self.pruned_update = update_idx
            if self.verbose:
                print(
                    '[hpo] '
                    f'trial={self.trial.number} '
                    f'pruned_at_update={update_idx}'
                )
            return False

        self._last_eval_timestep = int(self.num_timesteps)
        return True
