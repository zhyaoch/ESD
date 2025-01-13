from typing import Dict
import jax
import mujoco_py
import gym
import d4rl
import numpy as np
import jax.numpy as jnp
from collections import defaultdict
import time

def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """
    Wrapper that supplies a jax random key to a function (using keyword `seed`).
    Useful for stochastic policies that require randomness.

    Similar to functools.partial(f, seed=seed), but makes sure to use a different
    key for each new call (to avoid stale rng keys).

    """
    def wrapped(rng,*args, **kwargs): 
        return f(*args, seed=rng, **kwargs)

    return wrapped

def flatten(d, parent_key="", sep="."):
    """
    Helper function that flattens a dictionary of dictionaries into a single dictionary.
    E.g: flatten({'a': {'b': 1}}) -> {'a.b': 1}
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, "items"):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def kitchen_render(kitchen_env, wh=64):
    from dm_control.mujoco import engine
    camera = engine.MovableCamera(kitchen_env.sim, wh, wh)
    camera.set_pose(distance=1.8, lookat=[-0.3, .5, 2.], azimuth=90, elevation=-60)
    img = camera.render()
    return img


def evaluate_with_trajectories(
        policy_fn, high_policy_fn, policy_rep_fn, env: gym.Env, env_name, num_episodes: int, action_dim,observation_dim,seed,pretrain_dataset,base_observation=None, num_video_episodes=0,
        use_waypoints=False, eval_temperature=0, epsilon=0, goal_info=None,
        config=None
) -> Dict[str, float]: 
    trajectories = []
    stats = defaultdict(list)

    renders = []
    for i in range(num_episodes + num_video_episodes):
        trajectory = defaultdict(list)
            
        if 'humanoidmaze' in env_name or 'scene' in env_name or 'antsoccer' in env_name:
            observation, info = env.reset(options=dict(
                                    task_id=i%5+1,  # Set the evaluation task. Each environment provides five
                                                    # evaluation goals, and `task_id` must be in [1, 5].
                                    render_goal=False,  # Set to `True` to get a rendered goal image (optional).
                                ))
            done = False
        else:
            observation, done = env.reset(), False

        # Set goal
        if 'antmaze' in env_name:
            goal = env.wrapped_env.target_goal
            obs_goal = base_observation.copy()
            obs_goal[:2] = goal
        elif 'kitchen' in env_name:
            observation, obs_goal = observation[:30], observation[30:]
            obs_goal[:9] = base_observation[:9]
        elif 'calvin' in env_name:
            observation = observation['ob']
            goal = np.array([0.25, 0.15, 0, 0.088, 1, 1])
            obs_goal = base_observation.copy()
            obs_goal[15:21] = goal
        elif 'humanoidmaze' in env_name or 'scene' in env_name or 'antsoccer' in env_name:
            observation = observation
            # obs_goal = info['goal_rendered']
            obs_goal = info['goal']
        else:
            raise NotImplementedError

        render = []
        step = 0
        rng = seed
        while not done:
            rng, key = jax.random.split(rng)
            if not use_waypoints:
                cur_obs_goal = obs_goal
                if config['use_rep']:
                    cur_obs_goal_rep = policy_rep_fn(targets=cur_obs_goal, bases=observation)
                else:
                    cur_obs_goal_rep = cur_obs_goal
            else:
                cur_obs_goal = high_policy_fn(observations=observation, goals=obs_goal, temperature=eval_temperature,action_dim=observation_dim,rng=key)
                if config['use_rep']:
                    cur_obs_goal = cur_obs_goal / np.linalg.norm(cur_obs_goal, axis=-1, keepdims=True) * np.sqrt(cur_obs_goal.shape[-1])
                else:
                    cur_obs_goal = observation + cur_obs_goal
                cur_obs_goal_rep = cur_obs_goal

            rng, key = jax.random.split(rng)
            action = policy_fn(observations=observation, goals=cur_obs_goal_rep, low_dim_goals=True, temperature=eval_temperature,action_dim=action_dim,rng=key)
            action = pretrain_dataset.unnormalize(action)
            
            action = jnp.squeeze(action)
            if 'antmaze' in env_name:
                next_observation, r, done, info = env.step(action)
            elif 'kitchen' in env_name:
                next_observation, r, done, info = env.step(action)
                next_observation = next_observation[:30]
            elif 'calvin' in env_name:
                next_observation, r, done, info = env.step({'ac': np.array(action)})
                next_observation = next_observation['ob']
                del info['robot_info']
                del info['scene_info']
            elif 'humanoidmaze' in env_name or 'scene' in env_name or 'antsoccer' in env_name:  
                action = np.array(action)
                next_observation, r, terminated ,truncated, info = env.step(action)
                done = terminated or truncated
            step += 1
            if i >= num_episodes and step % 3 == 0:
                size = 768
                if 'antmaze' in env_name:
                    cur_frame = env.render(mode='rgb_array', width=size, height=size).transpose(2, 0, 1).copy()  #mode='rgb_array', 
                    if use_waypoints and not config['use_rep'] and ('large' in env_name or 'ultra' in env_name):
                        def xy_to_pixxy(x, y):
                            if 'large' in env_name:
                                pixx = (x / 36) * (0.93 - 0.07) + 0.07
                                pixy = (y / 24) * (0.21 - 0.79) + 0.79
                            elif 'ultra' in env_name:
                                pixx = (x / 52) * (0.955 - 0.05) + 0.05
                                pixy = (y / 36) * (0.19 - 0.81) + 0.81
                            return pixx, pixy
                        # print(f"cur_obs_goal_rep[:2]:{cur_obs_goal_rep[0,:2]}")
                        # x, y = cur_obs_goal_rep[:2]
                        x, y = cur_obs_goal_rep[0,:2]
                        pixx, pixy = xy_to_pixxy(x, y)
                        cur_frame[0, int((pixy - 0.02) * size):int((pixy + 0.02) * size), int((pixx - 0.02) * size):int((pixx + 0.02) * size)] = 255
                        cur_frame[1:3, int((pixy - 0.02) * size):int((pixy + 0.02) * size), int((pixx - 0.02) * size):int((pixx + 0.02) * size)] = 0
                    render.append(cur_frame)
                elif 'kitchen' in env_name:
                    render.append(kitchen_render(env, wh=1024).transpose(2, 0, 1))
                elif 'calvin' in env_name:
                    cur_frame = env.render(mode='rgb_array',wh=size).transpose(2, 0, 1)
                    render.append(cur_frame)
                elif 'humanoidmaze' in env_name or 'scene' in env_name or 'antsoccer' in env_name:
                    cur_frame = env.render().transpose(2, 0, 1)
                    render.append(cur_frame)
            transition = dict(
                observation=observation,
                next_observation=next_observation,
                action=action,
                reward=r,
                done=done,
                info=info,
            )
            add_to(trajectory, transition)
            add_to(stats, flatten(info))
            observation = next_observation
        if 'calvin' in env_name:
            info['return'] = sum(trajectory['reward'])
        add_to(stats, flatten(info, parent_key="final"))
        trajectories.append(trajectory)
        if i >= num_episodes: #zyc
            renders.append(np.array(render))

    for k, v in stats.items():
        stats[k] = np.mean(v)
    return stats, trajectories, renders

class EpisodeMonitor(gym.ActionWrapper):
    """A class that computes episode returns and lengths."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._reset_stats()
        self.total_timesteps = 0

    def _reset_stats(self):
        self.reward_sum = 0.0
        self.episode_length = 0
        self.start_time = time.time()

    def step(self, action: np.ndarray):
        observation, reward, done, info = self.env.step(action)

        self.reward_sum += reward
        self.episode_length += 1
        self.total_timesteps += 1
        info["total"] = {"timesteps": self.total_timesteps}

        if done:
            info["episode"] = {}
            info["episode"]["return"] = self.reward_sum
            info["episode"]["length"] = self.episode_length
            info["episode"]["duration"] = time.time() - self.start_time

            if hasattr(self, "get_normalized_score"):
                info["episode"]["normalized_return"] = (
                    self.get_normalized_score(info["episode"]["return"]) * 100.0
                )

        return observation, reward, done, info

    def reset(self) -> np.ndarray:
        self._reset_stats()
        return self.env.reset()
