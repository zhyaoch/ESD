from jaxrl_m.dataset import Dataset
from jaxrl_m.typing import *
from jaxrl_m.networks import *
import jax


class LayerNormMLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu
    activate_final: int = False
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_init()

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:                
                x = self.activations(x)
                x = nn.LayerNorm()(x)
        return x


class LayerNormRepresentation(nn.Module):
    hidden_dims: tuple = (256, 256)
    activate_final: bool = True
    ensemble: bool = True

    @nn.compact
    def __call__(self, observations):
        module = LayerNormMLP
        if self.ensemble:
            module = ensemblize(module, 2)
        return module(self.hidden_dims, activate_final=self.activate_final)(observations)


class Representation(nn.Module):
    hidden_dims: tuple = (256, 256)
    activate_final: bool = True
    ensemble: bool = True

    @nn.compact
    def __call__(self, observations):
        module = MLP
        if self.ensemble:
            module = ensemblize(module, 2)
        return module(self.hidden_dims, activate_final=self.activate_final, activations=nn.gelu)(observations)


class RelativeRepresentation(nn.Module):
    rep_dim: int = 256
    hidden_dims: tuple = (256, 256)
    module: nn.Module = None
    visual: bool = False
    layer_norm: bool = False
    rep_type: str = 'state'
    bottleneck: bool = True  # Meaning that we're using this representation for high-level actions

    @nn.compact
    def __call__(self, targets, bases=None):
        if bases is None:
            inputs = targets
        else:
            if self.rep_type == 'state':
                inputs = targets
            elif self.rep_type == 'diff':
                inputs = jax.tree_map(lambda t, b: t - b + jnp.ones_like(t) * 1e-6, targets, bases)
            elif self.rep_type == 'concat':
                inputs = jax.tree_map(lambda t, b: jnp.concatenate([t, b], axis=-1), targets, bases)
            else:
                raise NotImplementedError

        if self.visual:
            inputs = self.module()(inputs)
        if self.layer_norm:
            rep = LayerNormMLP(self.hidden_dims, activate_final=not self.bottleneck, activations=nn.gelu)(inputs)
        else:
            rep = MLP(self.hidden_dims, activate_final=not self.bottleneck, activations=nn.gelu)(inputs)

        if self.bottleneck:
            rep = rep / jnp.linalg.norm(rep, axis=-1, keepdims=True) * jnp.sqrt(self.rep_dim)
            # rep = rep / jnp.linalg.norm(rep, axis=-1, keepdims=True) * jnp.sqrt(1)
        return rep


class MonolithicVF(nn.Module):
    hidden_dims: tuple = (256, 256)
    readout_size: tuple = (256,)
    use_layer_norm: bool = True
    rep_dim: int = None
    obs_rep: int = 0

    def setup(self) -> None:
        repr_class = LayerNormRepresentation if self.use_layer_norm else Representation
        self.value_net = repr_class((*self.hidden_dims, 1), activate_final=False)

    def __call__(self, observations, goals=None, info=False):
        phi = observations
        psi = goals

        v1, v2 = self.value_net(jnp.concatenate([phi, psi], axis=-1)).squeeze(-1)

        if info:
            return {
                'v': (v1 + v2) / 2,
            }
        return v1, v2

class MonolithicVF_target(nn.Module):
    hidden_dims: tuple = (256, 256)
    readout_size: tuple = (256,)
    use_layer_norm: bool = True
    rep_dim: int = None
    obs_rep: int = 0

    def setup(self) -> None:
        repr_class = LayerNormRepresentation if self.use_layer_norm else Representation
        self.value_net = repr_class((*self.hidden_dims, 1), activate_final=False)

    def __call__(self, observations,targets, goals=None, info=False):
        phi = observations
        psi = goals

        q1, q2 = self.value_net(jnp.concatenate([phi,targets, psi], axis=-1)).squeeze(-1)

        if info:
            return {
                'q': (q1 + q2) / 2,
            }
        return q1, q2


def get_rep(
        encoder: nn.Module, targets: jnp.ndarray, bases: jnp.ndarray = None,
):
    if encoder is None:
        return targets
    else:
        if bases is None:
            return encoder(targets)
        else:
            return encoder(targets, bases)




class HierarchicalActorCritic(nn.Module):
    encoders: Dict[str, nn.Module]
    networks: Dict[str, nn.Module]
    use_waypoints: int

    def value(self, observations, goals, **kwargs):
        state_reps = get_rep(self.encoders['value_state'], targets=observations)
        goal_reps = get_rep(self.encoders['value_goal'], targets=goals, bases=observations)
        return self.networks['value'](state_reps, goal_reps, **kwargs)

    def target_value(self, observations, goals, **kwargs):
        state_reps = get_rep(self.encoders['value_state'], targets=observations)
        goal_reps = get_rep(self.encoders['value_goal'], targets=goals, bases=observations)
        return self.networks['target_value'](state_reps, goal_reps, **kwargs)
    
    def q_value(self, observations,targets, goals,low_dim_targets=False, **kwargs): 
        state_reps = get_rep(self.encoders['value_state'], targets=observations)
        if low_dim_targets:
            target_reps = targets
        else:
            target_reps = get_rep(self.encoders['value_goal'], targets=targets, bases=observations) 
        goal_reps = get_rep(self.encoders['value_goal'], targets=goals, bases=observations)
        return self.networks['q_value'](state_reps,target_reps, goal_reps, **kwargs)

    def target_q_value(self, observations,targets, goals,low_dim_targets=False, **kwargs): 
        state_reps = get_rep(self.encoders['value_state'], targets=observations)
        if low_dim_targets:
            target_reps = targets
        else:
            target_reps = get_rep(self.encoders['value_goal'], targets=targets, bases=observations) 
        goal_reps = get_rep(self.encoders['value_goal'], targets=goals, bases=observations)
        return self.networks['target_q_value'](state_reps,target_reps, goal_reps, **kwargs)

    def qa_value(self, observations,actions, goals,low_dim_goals=False, **kwargs): 
        state_reps = get_rep(self.encoders['value_state'], targets=observations)
        action_reps = get_rep(self.encoders['value_action'], targets=actions)
        if low_dim_goals:
            goal_reps = goals
        else:
            goal_reps = get_rep(self.encoders['value_goal'], targets=goals, bases=observations)
        return self.networks['qa_value'](state_reps,action_reps, goal_reps, **kwargs)

    def target_qa_value(self, observations,actions, goals,low_dim_goals=False, **kwargs):
        state_reps = get_rep(self.encoders['value_state'], targets=observations)
        action_reps = get_rep(self.encoders['value_action'], targets=actions)
        if low_dim_goals:
            goal_reps = goals
        else:
            goal_reps = get_rep(self.encoders['value_goal'], targets=goals, bases=observations)
        return self.networks['target_qa_value'](state_reps,action_reps, goal_reps, **kwargs)
    
    def actor(self, actions_noise,observations, goals,t, low_dim_goals=False, state_rep_grad=True, goal_rep_grad=True, **kwargs):
        state_reps = get_rep(self.encoders['policy_state'], targets=observations)
        time_reps = get_rep(self.encoders['time'], targets=t)
        if not state_rep_grad:
            state_reps = jax.lax.stop_gradient(state_reps)
            time_reps = jax.lax.stop_gradient(time_reps)

        if low_dim_goals:
            goal_reps = goals
        else:
            if self.use_waypoints:
                goal_reps = get_rep(self.encoders['value_goal'], targets=goals, bases=observations)
            else:
                goal_reps = get_rep(self.encoders['policy_goal'], targets=goals, bases=observations)
            if not goal_rep_grad:
                goal_reps = jax.lax.stop_gradient(goal_reps)

        return self.networks['actor'](jnp.concatenate([actions_noise,state_reps, goal_reps, time_reps], axis=-1), **kwargs)
    
    def high_actor(self,actions_noise, observations, goals,t, state_rep_grad=True, goal_rep_grad=True, **kwargs):
        state_reps = get_rep(self.encoders['high_policy_state'], targets=observations)
        time_reps = get_rep(self.encoders['high_time'], targets=t)
        if not state_rep_grad:
            state_reps = jax.lax.stop_gradient(state_reps)
            time_reps = jax.lax.stop_gradient(time_reps)

        goal_reps = get_rep(self.encoders['high_policy_goal'], targets=goals, bases=observations)
        if not goal_rep_grad:
            goal_reps = jax.lax.stop_gradient(goal_reps)

        return self.networks['high_actor'](jnp.concatenate([actions_noise,state_reps, goal_reps, time_reps], axis=-1), **kwargs)
    
    def value_goal_encoder(self, targets, bases, **kwargs):
        return get_rep(self.encoders['value_goal'], targets=targets, bases=bases)

    def policy_goal_encoder(self, targets, bases, **kwargs):
        assert not self.use_waypoints
        return get_rep(self.encoders['policy_goal'], targets=targets, bases=bases)

    def __call__(self, actions,observations_noise,observations, goals,targets,t):
        # Only for initialization
        rets = {
            'value': self.value(observations, goals),
            'target_value': self.target_value(observations, goals),
            'q_value': self.q_value(observations,targets, goals),
            'target_q_value': self.target_q_value(observations,targets, goals),
            'actor': self.actor(actions,observations, goals,t),
            'high_actor': self.high_actor(observations_noise,observations, goals,t),
            'qa_value': self.qa_value(observations,actions, goals), 
            'target_qa_value': self.target_qa_value(observations,actions, goals),
        }
        return rets

