"""Common networks used in RL.

This file contains nn.Module definitions for common networks used in RL. It is divided into three sets:

1) Common Networks: MLP
2) Common RL Networks:
    For discrete action spaces: DiscreteCritic is a Q-function
    For continuous action spaces: Critic, ValueCritic, and Policy provide the Q-function, value function, and policy respectively.
    For ensembling: ensemblize() provides a wrapper for creating ensembles of networks (e.g. for min-Q / double-Q)
3) Meta Networks for vision tasks:
    WithEncoder: Combines a fully connected network with an encoder network (encoder may come from jaxrl_m.vision)
    ActorCritic: Same as WithEncoder, but for possibly many different networks (e.g. actor, critic, value)
"""

from jaxrl_m.typing import *

import flax.linen as nn
import jax.numpy as jnp

import distrax
import flax.linen as nn
import jax.numpy as jnp

###############################
#
#  Common Networks
#
###############################


def default_init(scale: Optional[float] = 1.0):
    return nn.initializers.variance_scaling(scale, "fan_avg", "uniform")


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: int = False
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_init()

    def setup(self):
        self.layers = [
            nn.Dense(size, kernel_init=self.kernel_init) for size in self.hidden_dims
        ]

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i + 1 < len(self.layers) or self.activate_final:
                x = self.activations(x)
        return x


###############################
#
#
#  Common RL Networks
#
###############################


class DiscreteCritic(nn.Module):
    hidden_dims: Sequence[int]
    n_actions: int
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        return MLP((*self.hidden_dims, self.n_actions), activations=self.activations)(
            observations
        )


class Critic(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, actions], -1)
        critic = MLP((*self.hidden_dims, 1), activations=self.activations)(inputs)
        return jnp.squeeze(critic, -1)


def ensemblize(cls, num_qs, out_axes=0, **kwargs):
    """
    Useful for making ensembles of Q functions (e.g. double Q in SAC).

    Usage:

        critic_def = ensemblize(Critic, 2)(hidden_dims=hidden_dims)

    """
    return nn.vmap(
        cls,
        variable_axes={"params": 0},
        split_rngs={"params": True},
        in_axes=None,
        out_axes=out_axes,
        axis_size=num_qs,
        **kwargs
    )


class ValueCritic(nn.Module):
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        critic = MLP((*self.hidden_dims, 1))(observations)
        return jnp.squeeze(critic, -1)

class Policy(nn.Module):  ###new  非单位方差
    hidden_dims: Sequence[int]
    action_dim: int
    log_std_min: Optional[float] = -20
    log_std_max: Optional[float] = 2
    tanh_squash_distribution: bool = False
    state_dependent_std: bool = True
    final_fc_init_scale: float = 1e-2

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, temperature: float = 1.0
    ) -> distrax.Distribution:
        outputs = MLP(
            self.hidden_dims,
            activate_final=True,
        )(observations)

        means = nn.Dense(
            self.action_dim, kernel_init=default_init(self.final_fc_init_scale)
        )(outputs)
        if self.state_dependent_std:
            log_stds = nn.Dense(
                self.action_dim, kernel_init=default_init(self.final_fc_init_scale)
            )(outputs)
        else:
            log_stds = self.param("log_stds", nn.initializers.zeros, (self.action_dim,))

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(
            loc=means, scale_diag=jnp.exp(log_stds) * temperature
        )
        if self.tanh_squash_distribution:
            distribution = TransformedWithMode(
                distribution, distrax.Block(distrax.Tanh(), ndims=1)
            )
        return means,distribution


# class Policy(nn.Module):  ###new  单位方差
#     hidden_dims: Sequence[int]
#     action_dim: int
#     log_std_min: Optional[float] = -20
#     log_std_max: Optional[float] = 2
#     tanh_squash_distribution: bool = False
#     state_dependent_std: bool = True
#     final_fc_init_scale: float = 1e-2

#     @nn.compact
#     def __call__(
#         self, observations: jnp.ndarray, temperature: float = 1.0
#     ) -> distrax.Distribution:
#         outputs = MLP(
#             self.hidden_dims,
#             activate_final=True,
#         )(observations)

#         means = nn.Dense(
#             self.action_dim, kernel_init=default_init(self.final_fc_init_scale)
#         )(outputs)

#         distribution = distrax.MultivariateNormalDiag(
#             loc=means, scale_diag=jnp.ones(self.action_dim) * temperature
#         )
#         if self.tanh_squash_distribution:
#             distribution = TransformedWithMode(
#                 distribution, distrax.Block(distrax.Tanh(), ndims=1)
#             )
#         return means,distribution


def sinusoidal_embedding(timesteps, dim, max_period=20):  #总步长设置为20
  """
  Create sinusoidal timestep embeddings.
  :param timesteps: a 1-D Tensor of N indices, one per batch element.
                    These may be fractional.
  :param dim: the dimension of the output.
  :param max_period: controls the minimum frequency of the embeddings.
  :return: an [N x dim] Tensor of positional embeddings.
  """

  half = dim // 2
  freqs = jnp.exp(
    -jnp.log(max_period) * jnp.arange(half, dtype=jnp.float32) / half
  )
  # args = timesteps[:, None] * freqs[None, :]
  #args = jnp.expand_dims(timesteps, axis=-1) * freqs[None, :]
  args = timesteps * freqs
  embd = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
  return embd

def mish(x):
  return x * jnp.tanh(nn.softplus(x))

class TimeEmbedding(nn.Module):
  embed_size: int = 16
  act: callable = mish
  max_period: int =20

  @nn.compact
  def __call__(self, timesteps):
    x = sinusoidal_embedding(timesteps, self.embed_size,self.max_period)
    x = nn.Dense(self.embed_size * 2)(x)
    x = self.act(x)
    x = nn.Dense(self.embed_size)(x)
    return x


# class PolicyNet(nn.Module):
#   output_dim: int
#   arch: Tuple = (256, 256, 256)
#   time_embed_size: int = 16
#   act: callable = mish
#   use_layer_norm: bool = False

#   @nn.compact
#   def __call__(self, state, action, t):
#     time_embed = TimeEmbedding(self.time_embed_size, self.act)(t)
#     x = jnp.concatenate([state, action, time_embed], axis=-1)

#     for feat in self.arch:
#       x = nn.Dense(feat)(x)
#       if self.use_layer_norm:
#         x = nn.LayerNorm()(x)
#       x = self.act(x)

#     x = nn.Dense(self.output_dim)(x)
#     return x






class DiscretePolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    final_fc_init_scale: float = 1e-2

    @nn.compact
    def __call__(
            self, observations: jnp.ndarray, temperature: float = 1.0
    ) -> distrax.Distribution:
        outputs = MLP(
            self.hidden_dims,
            activate_final=True,
        )(observations)

        logits = nn.Dense(
            self.action_dim, kernel_init=default_init(self.final_fc_init_scale)
        )(outputs)

        distribution = distrax.Categorical(logits=logits / jnp.maximum(1e-6, temperature))

        return distribution


class TransformedWithMode(distrax.Transformed):
    def mode(self) -> jnp.ndarray:
        return self.bijector.forward(self.distribution.mode())


###############################
#
#
#   Meta Networks for Encoders
#
###############################


def get_latent(
    encoder: nn.Module, observations: Union[jnp.ndarray, Dict[str, jnp.ndarray]]
):
    """

    Get latent representation from encoder. If observations is a dict
        a state and image component, then concatenate the latents.

    """
    if encoder is None:
        return observations

    elif isinstance(observations, dict):
        return jnp.concatenate(
            [encoder(observations["image"]), observations["state"]], axis=-1
        )

    else:
        return encoder(observations)


class WithEncoder(nn.Module):
    encoder: nn.Module
    network: nn.Module

    def __call__(self, observations, *args, **kwargs):
        latents = get_latent(self.encoder, observations)
        return self.network(latents, *args, **kwargs)


class ActorCritic(nn.Module):
    """Combines FC networks with encoders for actor, critic, and value.

    Note: You can share encoder parameters between actor and critic by passing in the same encoder definition for both.

    Example:

        encoder_def = ImpalaEncoder()
        actor_def = Policy(...)
        critic_def = Critic(...)
        # This will share the encoder between actor and critic
        model_def = ActorCritic(
            encoders={'actor': encoder_def, 'critic': encoder_def},
            networks={'actor': actor_def, 'critic': critic_def}
        )
        # This will have separate encoders for actor and critic
        model_def = ActorCritic(
            encoders={'actor': encoder_def, 'critic': copy.deepcopy(encoder_def)},
            networks={'actor': actor_def, 'critic': critic_def}
        )
    """

    encoders: Dict[str, nn.Module]
    networks: Dict[str, nn.Module]

    def actor(self, observations, **kwargs):
        latents = get_latent(self.encoders["actor"], observations)
        return self.networks["actor"](latents, **kwargs)

    def critic(self, observations, actions, **kwargs):
        latents = get_latent(self.encoders["critic"], observations)
        return self.networks["critic"](latents, actions, **kwargs)

    def value(self, observations, **kwargs):
        latents = get_latent(self.encoders["value"], observations)
        return self.networks["value"](latents, **kwargs)

    def __call__(self, observations, actions):
        rets = {}
        if "actor" in self.networks:
            rets["actor"] = self.actor(observations)
        if "critic" in self.networks:
            rets["critic"] = self.critic(observations, actions)
        if "value" in self.networks:
            rets["value"] = self.value(observations)
        return rets
