from jaxrl_m.dataset import Dataset
from flax.core.frozen_dict import FrozenDict
from flax.core import freeze
import dataclasses
import numpy as np
import jax
import ml_collections

@dataclasses.dataclass
class GCDataset:
    dataset: Dataset
    p_randomgoal: float
    p_trajgoal: float
    p_currgoal: float
    geom_sample: int
    discount: float
    terminal_key: str = 'dones_float'
    reward_scale: float = 1.0
    reward_shift: float = -1.0
    terminal: bool = True

    @staticmethod
    def get_default_config():
        return ml_collections.ConfigDict({
            'p_randomgoal': 0.3,
            'p_trajgoal': 0.5,
            'p_currgoal': 0.2,
            'geom_sample': 0,
            'reward_scale': 1.0,
            'reward_shift': -1.0,
            'terminal': True,
        })
    def __post_init__(self):
        self.terminal_locs, = np.nonzero(self.dataset[self.terminal_key] > 0)
        self.discount_list = self.discount ** np.arange(self.way_steps) 
        self.normalize()
        assert np.isclose(self.p_randomgoal + self.p_trajgoal + self.p_currgoal, 1.0)

    def normalize(self):
        action_spec = self.dataset['actions']
        self._scale = np.max(action_spec) - np.min(action_spec)
        self._offset = np.min(action_spec)
        
        action_spec = (action_spec-self._offset)/self._scale
        action_spec = 2 * action_spec - 1  
        
        print(f"np.max(action_spec):{np.max(self.dataset['actions'])}")
        print(f"np.min(action_spec):{np.min(self.dataset['actions'])}")

    def unnormalize(self,action, eps=1e-4):
        action = (action + 1) / 2.
        return action * self._scale + self._offset
    
    
    def sample_goals(self, indx, p_randomgoal=None, p_trajgoal=None, p_currgoal=None):
        if p_randomgoal is None:
            p_randomgoal = self.p_randomgoal
        if p_trajgoal is None:
            p_trajgoal = self.p_trajgoal
        if p_currgoal is None:
            p_currgoal = self.p_currgoal

        batch_size = len(indx)
        # Random goals
        goal_indx = np.random.randint(self.dataset.size, size=batch_size)
        
        # Goals from the same trajectory
        final_state_indx = self.terminal_locs[np.searchsorted(self.terminal_locs, indx)]

        distance = np.random.rand(batch_size)
        if self.geom_sample:
            us = np.random.rand(batch_size)
            middle_goal_indx = np.minimum(indx + np.ceil(np.log(1 - us) / np.log(self.discount)).astype(int), final_state_indx)
        else:
            middle_goal_indx = np.round((np.minimum(indx + 1, final_state_indx) * distance + final_state_indx * (1 - distance))).astype(int)

        goal_indx = np.where(np.random.rand(batch_size) < p_trajgoal / (1.0 - p_currgoal), middle_goal_indx, goal_indx)
        
        # Goals at the current state
        goal_indx = np.where(np.random.rand(batch_size) < p_currgoal, indx, goal_indx)
        return goal_indx

    def sample(self, batch_size: int, indx=None):
        if indx is None:
            indx = np.random.randint(self.dataset.size-1, size=batch_size)
        
        batch = self.dataset.sample(batch_size, indx)
        goal_indx = self.sample_goals(indx)

        success = (indx == goal_indx)
        batch['rewards'] = success.astype(float) * self.reward_scale + self.reward_shift
        if self.terminal:
            batch['masks'] = (1.0 - success.astype(float))
        else:
            batch['masks'] = np.ones(batch_size)
        batch['goals'] = jax.tree_map(lambda arr: arr[goal_indx], self.dataset['observations'])

        return batch

@dataclasses.dataclass
class GCSDataset(GCDataset):
    way_steps: int = None
    high_p_randomgoal: float = 0.

    @staticmethod
    def get_default_config():
        return ml_collections.ConfigDict({
            'p_randomgoal': 0.3,
            'p_trajgoal': 0.5,
            'p_currgoal': 0.2,
            'geom_sample': 0,
            'reward_scale': 1.0,
            'reward_shift': -1.0,  #0.0
            'terminal': False,
        })
        
    def slice_rewards(self,arr, start_idxs, end_idxs):
        return np.array([self.reward_shift*np.sum(self.discount_list[:end-start]) for start, end in zip(start_idxs, end_idxs)])

    def sample(self, batch_size: int, indx=None):
        if indx is None:
            indx = np.random.randint(self.dataset.size-1, size=batch_size)

        batch = self.dataset.sample(batch_size, indx)
        goal_indx = self.sample_goals(indx)
        target_indx = np.random.randint(self.way_steps//2,self.way_steps, size=batch_size) 
        
        success = (indx == goal_indx)

        batch['rewards'] = success.astype(float) * self.reward_scale + self.reward_shift

        if self.terminal:
            batch['masks'] = (1.0 - success.astype(float))
        else:
            batch['masks'] = np.ones(batch_size)

        batch['goals'] = jax.tree_map(lambda arr: arr[goal_indx], self.dataset['observations'])

        final_state_indx = self.terminal_locs[np.searchsorted(self.terminal_locs, indx)]
        way_indx = np.minimum(indx + target_indx, final_state_indx)
        batch['low_goals'] = jax.tree_map(lambda arr: arr[way_indx], self.dataset['observations'])

        distance = np.random.rand(batch_size)
        
        
        high_traj_goal_indx = np.round((np.minimum(indx + 1, final_state_indx) * distance + final_state_indx * (1 - distance))).astype(int)
        high_traj_target_indx = np.minimum(indx + target_indx, high_traj_goal_indx)

        high_random_goal_indx = np.random.randint(self.dataset.size, size=batch_size)
        high_random_target_indx = np.minimum(indx + target_indx, final_state_indx)

        pick_random = (np.random.rand(batch_size) < self.high_p_randomgoal)
        high_goal_idx = np.where(pick_random, high_random_goal_indx, high_traj_goal_indx)
        high_target_idx = np.where(pick_random, high_random_target_indx, high_traj_target_indx)

        batch['high_goals'] = jax.tree_map(lambda arr: arr[high_goal_idx], self.dataset['observations'])
        batch['high_targets'] = jax.tree_map(lambda arr: arr[high_target_idx], self.dataset['observations'])

        if isinstance(batch['goals'], FrozenDict):
            # Freeze the other observations
            batch['observations'] = freeze(batch['observations'])
            batch['next_observations'] = freeze(batch['next_observations'])
        
        
        batch['observations_targets']=np.copy(batch['observations'])
        batch['next_observations_targets']=np.copy(batch['next_observations'])
        batch['action_targets']=np.copy(batch['actions'])


        success = (((high_target_idx-1) == goal_indx) | (indx == goal_indx))
        batch['rewards_targets'] = success.astype(float) * self.reward_scale + self.reward_shift
        batch['rewards_sum'] = jax.tree_map(lambda arr: self.slice_rewards(arr, indx, high_target_idx), self.dataset['rewards'])
        batch['rewards_sum'] = batch['rewards_sum']*np.abs(batch['rewards']) 
        batch['distence']=high_target_idx-indx
        
        return batch

