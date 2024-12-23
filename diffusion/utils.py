
import jax.numpy as jnp
import numpy as np
import jax 
import math
from PIL import Image
import wandb
from ml_collections import ConfigDict


def cosine_beta_schedule(timesteps):
    """Return cosine schedule 
    as proposed in https://arxiv.org/abs/2102.09672 """
    s=0.008
    max_beta=0.999
    ts = jnp.linspace(0, 1, timesteps + 1)
    alphas_bar = jnp.cos((ts + s) / (1 + s) * jnp.pi /2) ** 2
    alphas_bar = alphas_bar/alphas_bar[0]
    betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
    return(jnp.clip(betas, 0, max_beta))

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    betas = jnp.linspace(
        beta_start, beta_end, timesteps, dtype=jnp.float64)
    return(betas)

def get_ddpm_params(config):
    schedule_name = config.beta_schedule
    timesteps = config.timesteps
    p2_loss_weight_gamma = config.p2_loss_weight_gamma
    p2_loss_weight_k = config.p2_loss_weight_gamma

    if schedule_name == 'linear':
        betas = linear_beta_schedule(timesteps)
    elif schedule_name == 'cosine':
        betas = cosine_beta_schedule(timesteps)
    else:
        raise ValueError(f'unknown beta schedule {schedule_name}')
    assert betas.shape == (timesteps,)
    alphas = 1. - betas
    alphas_bar = jnp.cumprod(alphas, axis=0)
    sqrt_alphas_bar = jnp.sqrt(alphas_bar)
    sqrt_1m_alphas_bar= jnp.sqrt(1. - alphas_bar)
    
    # calculate p2 reweighting
    p2_loss_weight=  (p2_loss_weight_k + alphas_bar / (1 - alphas_bar)) ** -p2_loss_weight_gamma

    return {
      'betas': betas,
      'alphas': alphas,
      'alphas_bar': alphas_bar,
      'sqrt_alphas_bar': sqrt_alphas_bar,
      'sqrt_1m_alphas_bar': sqrt_1m_alphas_bar,
      'p2_loss_weight': p2_loss_weight
  }

def q_sample(x, t, noise, ddpm_params):
    sqrt_alpha_bar = ddpm_params['sqrt_alphas_bar'][t, None]
    sqrt_1m_alpha_bar = ddpm_params['sqrt_1m_alphas_bar'][t,None]
    x_t = sqrt_alpha_bar * x + sqrt_1m_alpha_bar * noise
    return x_t


def noise_to_x0(noise, xt, batched_t, ddpm):
    assert batched_t.shape[0] == xt.shape[0] == noise.shape[0]
    sqrt_alpha_bar = ddpm['sqrt_alphas_bar'][batched_t] ###
    alpha_bar= ddpm['alphas_bar'][batched_t] ###
    x0 = 1. / sqrt_alpha_bar * xt -  jnp.sqrt(1./alpha_bar-1) * noise
    return x0


def x0_to_noise(x0, xt, batched_t, ddpm):
    assert batched_t.shape[0] == xt.shape[0] == x0.shape[0]
    sqrt_alpha_bar = ddpm['sqrt_alphas_bar'][batched_t] ###
    alpha_bar= ddpm['alphas_bar'][batched_t] ###
    noise = (1. / sqrt_alpha_bar * xt - x0) /jnp.sqrt(1./alpha_bar-1)
    return noise


def get_posterior_mean_variance(x, t, x0, v, ddpm_params):

    beta = ddpm_params['betas'][t+1]
    alpha = ddpm_params['alphas'][t+1]
    alpha_bar = ddpm_params['alphas_bar'][t+1]
    alpha_bar_last = ddpm_params['alphas_bar'][t]
    sqrt_alpha_bar_last = ddpm_params['sqrt_alphas_bar'][t]

    coef_x0 = beta * sqrt_alpha_bar_last / (1. - alpha_bar)
    coef_xt = (1. - alpha_bar_last) * jnp.sqrt(alpha) / ( 1- alpha_bar)        
    posterior_mean = coef_x0 * x0 + coef_xt * x
        
    posterior_variance = beta * (1 - alpha_bar_last) / (1. - alpha_bar)
    posterior_log_variance = jnp.log(jnp.clip(posterior_variance, a_min = 1e-20))

    return posterior_mean, posterior_log_variance