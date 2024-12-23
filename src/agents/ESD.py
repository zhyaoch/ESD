import copy

from jaxrl_m.typing import *

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxrl_m.common import TrainState, target_update
from jaxrl_m.networks import Policy, Critic, ensemblize, DiscretePolicy,TimeEmbedding

import flax
import flax.linen as nn
from flax.core import freeze, unfreeze
import ml_collections
from . import iql
from src.special_networks import Representation, HierarchicalActorCritic, RelativeRepresentation, MonolithicVF,MonolithicVF_target
from diffusion import utils,q_sample,noise_to_x0,x0_to_noise,get_posterior_mean_variance
import time
import os
import flax
import jax.numpy as jnp
import numpy as np
import jax 
from flax import jax_utils

ddpm = ml_collections.ConfigDict()
ddpm.beta_schedule = 'linear'
ddpm.timesteps = 20  
ddpm.p2_loss_weight_gamma = 0. 
ddpm.p2_loss_weight_k = 1
ddpm.self_condition = False 
ddpm.pred_x0 = True 
ddpm_params = utils.get_ddpm_params(ddpm) 

def flatten(x):
  return x.reshape(x.shape[0], -1)

def l2_loss(logit, target):
    return (logit - target)**2

def expectile_loss(adv, diff, expectile=0.7):  
    weight = jnp.where(adv >= 0, expectile, (1 - expectile))
    return weight * (diff**2)


def compute_actor_loss(agent, batch, network_params, rng: PRNGKey, is_pred_x0=True):
    if agent.config['use_waypoints']:  
        cur_goals = batch['low_goals']
    else:  
        cur_goals = batch['high_goals']
    v1, v2 = agent.network(batch['observations'], cur_goals, method='value')
    v = (v1 + v2) / 2
    nv1, nv2 = agent.network(batch['next_observations'], cur_goals, method='value')
    nv = (nv1 + nv2) / 2

    adv = nv - v
    exp_a = jnp.exp(adv * agent.config['temperature'])
    exp_a = jnp.minimum(exp_a, 100.0)

    if agent.config['use_waypoints']:
        goal_rep_grad = agent.config['policy_train_rep']
    else:
        goal_rep_grad = True
    
    x = batch['actions']
    B, C = x.shape
    rng, t_rng = jax.random.split(rng)
    batched_t = jax.random.randint(t_rng, shape=(B,), dtype = jnp.int32, minval=0, maxval= len(ddpm_params['betas']))

    rng, noise_rng = jax.random.split(rng)
    noise = jax.random.normal(noise_rng, x.shape)
    target = x if is_pred_x0 else noise
    x_t = q_sample(x, batched_t, noise, ddpm_params)
    
    p2_loss_weight = ddpm_params['p2_loss_weight']

    pred,dist = agent.network(x_t,batch['observations'], cur_goals,batched_t[:,None], state_rep_grad=True, goal_rep_grad=goal_rep_grad, method='actor', params=network_params) ###new
    loss = l2_loss(flatten(pred),flatten(target))
    loss = jnp.mean(loss, axis= 1)
    assert loss.shape == (B,)
    loss = loss * p2_loss_weight[batched_t]
    actor_loss = (exp_a*loss).mean()
    return actor_loss, {
        'actor_loss': actor_loss,
        'adv': adv.mean(),
        'adv_median': jnp.median(adv),
        'diff_loss': loss.mean()
    }


def compute_high_actor_loss(agent, batch, network_params, rng: PRNGKey, is_pred_x0=True):
    cur_goals = batch['high_goals']
    v1, v2 = agent.network(batch['observations_targets'],cur_goals, method='value')
    (q1_t, q2_t) = agent.network(batch['observations_targets'], batch['high_targets'],cur_goals, method='q_value')
    q_t = (q1_t + q2_t) / 2
    v = (v1 + v2) / 2

    adv = q_t - v
    exp_a = jnp.exp(adv * agent.config['high_temperature'])
    exp_a = jnp.minimum(exp_a, 100.0)

    if agent.config['use_rep']:
        target = agent.network(targets=batch['high_targets'], bases=batch['observations_targets'], method='value_goal_encoder')
    else:
        target = batch['high_targets'] - batch['observations_targets']
    
    x= target
    B, C = target.shape
    rng, t_rng = jax.random.split(rng)
    batched_t = jax.random.randint(t_rng, shape=(B,), dtype = jnp.int32, minval=0, maxval= len(ddpm_params['betas']))
   
    rng, noise_rng = jax.random.split(rng)
    noise = jax.random.normal(noise_rng, x.shape)
    target = x if is_pred_x0 else noise
    x_t = q_sample(x, batched_t, noise, ddpm_params)
    
    pred,dist = agent.network(x_t,batch['observations'], cur_goals,batched_t[:,None], state_rep_grad=True, goal_rep_grad=True, method='high_actor', params=network_params)
    p2_loss_weight = ddpm_params['p2_loss_weight']
    loss = l2_loss(flatten(pred),flatten(target))
    loss = jnp.mean(loss, axis= 1)
    assert loss.shape == (B,)
    loss = loss * p2_loss_weight[batched_t]
    actor_loss = (exp_a*loss).mean()


    return actor_loss, {
        'high_actor_loss': actor_loss,
        'high_adv': adv.mean(),
        'high_adv_median': jnp.median(adv),
        'diff_loss': loss.mean(),
    }


def compute_value_loss(agent, batch, network_params):
    # masks are 0 if terminal, 1 otherwise
    batch['masks'] = - batch['rewards']
    (next_v1, next_v2) = agent.network(batch['next_observations'], batch['goals'], method='target_value')
    next_v = jnp.minimum(next_v1, next_v2)
    q = batch['rewards'] + agent.config['discount'] * batch['masks'] * next_v

    (v1_t, v2_t) = agent.network(batch['observations'], batch['goals'], method='target_value')
    v_t = (v1_t + v2_t) / 2
    adv = q - v_t

    q1 = batch['rewards'] + agent.config['discount'] * batch['masks'] * next_v1
    q2 = batch['rewards'] + agent.config['discount'] * batch['masks'] * next_v2
    (v1, v2) = agent.network(batch['observations'], batch['goals'], method='value', params=network_params)

    value_loss1 = expectile_loss(adv, q1 - v1, agent.config['pretrain_expectile']).mean()
    value_loss2 = expectile_loss(adv, q2 - v2, agent.config['pretrain_expectile']).mean()
    value_loss = value_loss1 + value_loss2

    advantage = adv
    return value_loss, {
        'value_loss': value_loss,
        'v max': v1.max(),
        'v min': v1.min(),
        'v mean': v1.mean(),
        'abs adv mean': jnp.abs(advantage).mean(),
        'adv mean': advantage.mean(),
        'adv max': advantage.max(),
        'adv min': advantage.min(),
        'accept prob': (advantage >= 0).mean(),
    }


def compute_q_value_loss(agent, batch, network_params): 
    # masks are 0 if terminal, 1 otherwise
    batch['masks'] = - batch['rewards_targets']
    (next_v_1, next_v_2) = agent.network(batch['high_targets'], batch['goals'], method='target_value')
    next_v = jnp.minimum(next_v_1, next_v_2)
    q = batch['rewards_sum'] + (agent.config['discount']**batch['distence']) * batch['masks'] * next_v

    (q1_t, q2_t) = agent.network(batch['observations_targets'], batch['high_targets'], batch['goals'], method='target_q_value')
    q_t = (q1_t + q2_t) / 2
    adv = q - q_t

    q1 = batch['rewards_sum'] + (agent.config['discount']**batch['distence']) * batch['masks'] * next_v_1
    q2 = batch['rewards_sum'] + (agent.config['discount']**batch['distence']) * batch['masks'] * next_v_2
    (q_1, q_2) = agent.network(batch['observations_targets'], batch['high_targets'], batch['goals'], method='q_value', params=network_params)

    q_value_loss1 = expectile_loss(adv, q1 - q_1, agent.config['pretrain_expectile']).mean()
    q_value_loss2 = expectile_loss(adv, q2 - q_2, agent.config['pretrain_expectile']).mean()
    q_value_loss = q_value_loss1 + q_value_loss2

    advantage = adv
    return q_value_loss, {
        'q_value_loss': q_value_loss,
        'q max': q_1.max(),
        'q min': q_1.min(),
        'q mean': q_1.mean(),
        'abs adv mean': jnp.abs(advantage).mean(),
        'adv mean': advantage.mean(),
        'adv max': advantage.max(),
        'adv min': advantage.min(),
        'accept prob': (advantage >= 0).mean(),
    }
    

def compute_qa_value_loss(agent, batch, network_params):  #ESD
    # masks are 0 if terminal, 1 otherwise
    batch['masks'] = - batch['rewards']
    (next_v1, next_v2) = agent.network(batch['next_observations'], batch['goals'], method='target_value')
    q1 = batch['rewards'] + agent.config['discount'] * batch['masks'] * next_v1
    q2 = batch['rewards'] + agent.config['discount'] * batch['masks'] * next_v2
    (qa1, qa2) = agent.network(batch['observations'], batch['actions'], batch['goals'], method='qa_value', params=network_params)

    value_loss1 = ((q1 - qa1)**2).mean()
    value_loss2 = ((q2 - qa2)**2).mean()
    value_loss = value_loss1 + value_loss2
    
    return value_loss, {
        'value_loss': value_loss,
        'q max': qa1.max(),
        'q min': qa1.min(),
        'q mean': qa1.mean(),
    }

class JointTrainAgent(iql.IQLAgent):
    network: TrainState = None

    def pretrain_update(agent, pretrain_batch, seed: PRNGKey, value_update=True, actor_update=True, high_actor_update=True, is_pred_x0=ddpm.pred_x0,qa_value_update=False):
        def loss_fn(network_params):
            info = {}

            # Value
            if value_update:
                value_loss, value_info = compute_value_loss(agent, pretrain_batch, network_params)
                for k, v in value_info.items():
                    info[f'value/{k}'] = v
            else:
                value_loss = 0.
                
            # qs_Value
            if value_update:
                q_value_loss, q_value_info = compute_q_value_loss(agent, pretrain_batch, network_params)
                for k, v in q_value_info.items():
                    info[f'q_value/{k}'] = v
            else:
                q_value_loss = 0.
            rng, update_rng = jax.random.split(seed)
            
            # qa_Value  #ESD
            if qa_value_update: 
                qa_value_loss, qa_value_info = compute_qa_value_loss(agent, pretrain_batch, network_params)
                for k, v in qa_value_info.items():
                    info[f'qa_value/{k}'] = v
            else:
                qa_value_loss = 0.

            # Actor
            if actor_update:
                actor_loss, actor_info = compute_actor_loss(agent, pretrain_batch, network_params,rng=update_rng,is_pred_x0=is_pred_x0)
                for k, v in actor_info.items():
                    info[f'actor/{k}'] = v
            else:
                actor_loss = 0.

            rng, update_rng = jax.random.split(rng)
            # High Actor
            if high_actor_update and agent.config['use_waypoints']:
                high_actor_loss, high_actor_info = compute_high_actor_loss(agent, pretrain_batch, network_params,rng=update_rng,is_pred_x0=is_pred_x0)
                for k, v in high_actor_info.items():
                    info[f'high_actor/{k}'] = v
            else:
                high_actor_loss = 0.

            loss =q_value_loss + value_loss + actor_loss + high_actor_loss + qa_value_loss

            return loss, info

        if value_update:
            new_target_params = jax.tree_map(
                lambda p, tp: p * agent.config['target_update_rate'] + tp * (1 - agent.config['target_update_rate']), agent.network.params['networks_value'], agent.network.params['networks_target_value']
            )
            
            new_q_target_params = jax.tree_map(
                lambda p, tp: p * agent.config['target_update_rate'] + tp * (1 - agent.config['target_update_rate']), agent.network.params['networks_q_value'], agent.network.params['networks_target_q_value']
            )

            new_qa_target_params = jax.tree_map(
                lambda p, tp: p * agent.config['target_update_rate'] + tp * (1 - agent.config['target_update_rate']), agent.network.params['networks_qa_value'], agent.network.params['networks_target_qa_value'] 
            )

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn, has_aux=True)

        if value_update:
            params = unfreeze(new_network.params)
            params['networks_target_value'] = new_target_params
            params['networks_target_q_value'] = new_q_target_params
            params['networks_target_qa_value'] = new_qa_target_params
            new_network = new_network.replace(params=freeze(params))

        return agent.replace(network=new_network), info
    pretrain_update = jax.jit(pretrain_update, static_argnames=('value_update', 'actor_update', 'high_actor_update',"is_pred_x0")) 

    def sample_high_actions(agent,
                            observations: np.ndarray,
                            goals: np.ndarray,
                            *,
                            action_dim: int,
                            seed: PRNGKey,
                            temperature: float = 1.0,
                            num_samples: int = 8) -> jnp.ndarray:
        def model_predict(x, x0,observation, cur_goals, t, ddpm_params, self_condition, is_pred_x0=True, use_ema=True):
            pred,dist  = agent.network(x,observation, cur_goals,t, temperature=temperature, method='high_actor')
            if is_pred_x0: 
                x0_pred = pred
                noise_pred =  x0_to_noise(pred, x, t, ddpm_params)
            else:
                noise_pred = pred
                x0_pred = noise_to_x0(pred, x, t, ddpm_params)
            
            return x0_pred, noise_pred

        def ddpm_sample_step(rng, x,observation, cur_goals, t, x0_last, ddpm_params, self_condition=False, is_pred_x0=True):
            batched_t = jnp.ones((x.shape[0],1), dtype=jnp.int32) * t
            x0, v = model_predict(x, None,observation, cur_goals, batched_t,ddpm_params, self_condition, is_pred_x0, use_ema=True)
            posterior_mean, posterior_log_variance = get_posterior_mean_variance(x, t, x0, v, ddpm_params)
            scalar_value = 1 - (t == 0).astype(float) 
            nonzero_mask = jnp.full_like(x, scalar_value)
            x = posterior_mean + nonzero_mask*jnp.exp(0.5 *  posterior_log_variance) * jax.random.normal(rng, x.shape)*0.5 
            return x, x0
        

        rng, x_rng = jax.random.split(seed)
        list_x0 = []
        if num_samples is None:
            x = jax.random.normal(x_rng, shape=[1,action_dim])
        else:
            x = jax.random.normal(x_rng, shape=[num_samples,action_dim])
        x0 = jnp.zeros_like(x) 
        
        t_values = np.arange(ddpm.timesteps)
        replicated_t_values = [jax_utils.replicate(t) for t in reversed(t_values)]
        for t in replicated_t_values:
            rng, step_rng = jax.random.split(rng)
            x, x0 = ddpm_sample_step(step_rng, x,jnp.tile(observations, (num_samples, 1)),jnp.tile(goals, (num_samples, 1)), t, x0, ddpm_params=ddpm_params, self_condition=False, is_pred_x0=ddpm.pred_x0)
            list_x0.append(x0)
        if num_samples>1:
            (q1, q2)=agent.network(jnp.tile(observations, (num_samples, 1)),x,jnp.tile(goals, (num_samples, 1)),low_dim_targets=True, method='target_q_value')
            q= q1+q2
            max_index = jnp.argmax(q)
            x = x[max_index,:][None,:]
        return x
    sample_high_actions = jax.jit(sample_high_actions, static_argnames=('action_dim', 'num_samples'))
    
    def sample_actions(agent,
                       observations: np.ndarray,
                       goals: np.ndarray,
                       *,
                       action_dim: int,
                       low_dim_goals: bool = False,
                       seed: PRNGKey,
                       temperature: float = 1.0,
                       discrete: int = 0,
                       num_samples: int = 1) -> jnp.ndarray:
        def model_predict(x, x0,observation, cur_goals, t, ddpm_params, self_condition, is_pred_x0=True, use_ema=True):
            pred,dist = agent.network(x,observation, cur_goals,t, low_dim_goals=low_dim_goals, temperature=temperature, method='actor')
            if is_pred_x0: 
                x0_pred = pred
                noise_pred =  x0_to_noise(pred, x, t, ddpm_params)
            else:
                noise_pred = pred
                x0_pred = noise_to_x0(pred, x, t, ddpm_params)
            
            return x0_pred, noise_pred

        def ddpm_sample_step(rng, x,observation, cur_goals, t, x0_last, ddpm_params, self_condition=False, is_pred_x0=True):
            batched_t = jnp.ones((x.shape[0],1), dtype=jnp.int32) * t
            x0, v = model_predict(x, None,observation, cur_goals, batched_t,ddpm_params, self_condition, is_pred_x0, use_ema=True)
            x0 = jnp.clip(x0, -1., 1.)
            
            scalar_value = 1 - (t == 0).astype(float)
            nonzero_mask = jnp.full_like(x, scalar_value)
            posterior_mean, posterior_log_variance = get_posterior_mean_variance(x, t, x0, v, ddpm_params)
            x = posterior_mean + nonzero_mask*jnp.exp(0.5 *  posterior_log_variance) * jax.random.normal(rng, x.shape)*0.5 #噪声强度设置为0.5
            return x, x0
        
        rng, x_rng = jax.random.split(seed)
        list_x0 = []
        if num_samples is None:
            x = jax.random.normal(x_rng, shape=[1,action_dim])
        else:
            x = jax.random.normal(x_rng, shape=[num_samples,action_dim])
        x0 = jnp.zeros_like(x) 
        t_values = np.arange(ddpm.timesteps)
        replicated_t_values = [jax_utils.replicate(t) for t in reversed(t_values)]
        for t in replicated_t_values:
            rng, step_rng = jax.random.split(rng)
            x, x0 = ddpm_sample_step(step_rng, x,jnp.tile(observations, (num_samples, 1)),jnp.tile(goals, (num_samples, 1)), t, x0, ddpm_params=ddpm_params, self_condition=False, is_pred_x0=ddpm.pred_x0)
            list_x0.append(x0)
        if num_samples>1:
            (q1, q2)=agent.network(jnp.tile(observations, (num_samples, 1)),x,jnp.tile(goals, (num_samples, 1)),low_dim_goals=True, method='target_qa_value')
            q= q1+q2
            max_index = jnp.argmax(q)
            x = x[max_index,:][None,:]
        return x
    sample_actions = jax.jit(sample_actions, static_argnames=('action_dim', 'num_samples', 'low_dim_goals', 'discrete'))


    @jax.jit
    def get_policy_rep(agent,
                       *,
                       targets: np.ndarray,
                       bases: np.ndarray = None,
                       ) -> jnp.ndarray:
        return agent.network(targets=targets, bases=bases, method='policy_goal_encoder')


def create_learner(
        seed: int,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        lr: float = 3e-4,
        actor_hidden_dims: Sequence[int] = (256,256, 256),
        value_hidden_dims: Sequence[int] = (256, 256),
        discount: float = 0.99,
        tau: float = 0.005,
        temperature: float = 1,
        high_temperature: float = 1,
        pretrain_expectile: float = 0.7,
        way_steps: int = 0,
        rep_dim: int = 10,
        use_rep: int = 0,
        policy_train_rep: float = 0,
        visual: int = 0,
        encoder: str = 'impala',
        discrete: int = 0,
        use_layer_norm: int = 0,
        rep_type: str = 'state',
        use_waypoints: int = 0,
        **kwargs):

        print('Extra kwargs:', kwargs)

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, high_actor_key, critic_key, value_key = jax.random.split(rng, 5)

        value_state_encoder = None
        value_goal_encoder = None
        policy_state_encoder = None
        policy_goal_encoder = None
        high_policy_state_encoder = None
        high_policy_goal_encoder = None
        time_encoder = TimeEmbedding(max_period=ddpm.timesteps)
        high_time_encoder = TimeEmbedding(max_period=ddpm.timesteps)
        value_action_encoder = None
        if visual:
            assert use_rep
            from jaxrl_m.vision import encoders

            visual_encoder = encoders[encoder]
            def make_encoder(bottleneck):
                if bottleneck:
                    return RelativeRepresentation(rep_dim=rep_dim, hidden_dims=(rep_dim,), visual=True, module=visual_encoder, layer_norm=use_layer_norm, rep_type=rep_type, bottleneck=True)
                else:
                    return RelativeRepresentation(rep_dim=value_hidden_dims[-1], hidden_dims=(value_hidden_dims[-1],), visual=True, module=visual_encoder, layer_norm=use_layer_norm, rep_type=rep_type, bottleneck=False)

            value_state_encoder = make_encoder(bottleneck=False)
            value_goal_encoder = make_encoder(bottleneck=use_waypoints)
            policy_state_encoder = make_encoder(bottleneck=False)
            policy_goal_encoder = make_encoder(bottleneck=False)
            high_policy_state_encoder = make_encoder(bottleneck=False)
            high_policy_goal_encoder = make_encoder(bottleneck=False)
        else:
            def make_encoder(bottleneck):
                if bottleneck:
                    return RelativeRepresentation(rep_dim=rep_dim, hidden_dims=(*value_hidden_dims, rep_dim), layer_norm=use_layer_norm, rep_type=rep_type, bottleneck=True)
                else:
                    return RelativeRepresentation(rep_dim=value_hidden_dims[-1], hidden_dims=(*value_hidden_dims, value_hidden_dims[-1]), layer_norm=use_layer_norm, rep_type=rep_type, bottleneck=False)

            if use_rep:
                value_goal_encoder = make_encoder(bottleneck=True)

        value_def = MonolithicVF(hidden_dims=value_hidden_dims, use_layer_norm=use_layer_norm, rep_dim=rep_dim)
        q_value_def = MonolithicVF_target(hidden_dims=value_hidden_dims, use_layer_norm=use_layer_norm, rep_dim=rep_dim)
        qa_value_def = MonolithicVF_target(hidden_dims=value_hidden_dims, use_layer_norm=use_layer_norm, rep_dim=rep_dim) #ESD
        
        if discrete:
            action_dim = actions[0] + 1
            actor_def = DiscretePolicy(actor_hidden_dims, action_dim=action_dim)
        else:
            action_dim = actions.shape[-1]
            actor_def = Policy(actor_hidden_dims, action_dim=action_dim)

        high_action_dim = observations.shape[-1] if not use_rep else rep_dim
        high_actor_def = Policy(actor_hidden_dims, action_dim=high_action_dim)

        network_def = HierarchicalActorCritic(  
            encoders={
                'value_state': value_state_encoder,
                'value_goal': value_goal_encoder,
                'policy_state': policy_state_encoder,
                'policy_goal': policy_goal_encoder,
                'high_policy_state': high_policy_state_encoder,
                'high_policy_goal': high_policy_goal_encoder,
                'time': time_encoder,
                'high_time': high_time_encoder,
                'value_action': value_action_encoder,
            },
            networks={
                'value': value_def,
                'target_value': copy.deepcopy(value_def),
                'q_value': q_value_def,
                'target_q_value': copy.deepcopy(q_value_def),
                'actor': actor_def,
                'high_actor': high_actor_def,
                'qa_value': qa_value_def, 
                'target_qa_value': copy.deepcopy(qa_value_def), 
            },
            use_waypoints=use_waypoints,
        )
        N = observations.shape[0]
        network_tx = optax.chain(optax.zero_nans(), optax.adam(learning_rate=lr))
        network_params = network_def.init(value_key, actions,observations[:,:high_action_dim],observations, observations,observations,np.zeros((N , 1)))['params'] 
        network = TrainState.create(network_def, network_params, tx=network_tx)
        params = unfreeze(network.params)
        params['networks_target_value'] = params['networks_value']
        network = network.replace(params=freeze(params))

        config = flax.core.FrozenDict(dict(
            discount=discount, temperature=temperature, high_temperature=high_temperature,
            target_update_rate=tau, pretrain_expectile=pretrain_expectile, way_steps=way_steps, rep_dim=rep_dim,
            policy_train_rep=policy_train_rep,
            use_rep=use_rep, use_waypoints=use_waypoints,
        ))

        return JointTrainAgent(rng, network=network, critic=None, value=None, target_value=None, actor=None, config=config)


def get_default_config():
    config = ml_collections.ConfigDict({
        'lr': 3e-4,
        'actor_hidden_dims': (512,512,512),
        'value_hidden_dims': (256, 256),
        'discount': 0.99,
        'temperature': 1.0,
        'tau': 0.005,
        'pretrain_expectile': 0.7,
    })

    return config