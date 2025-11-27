
from __future__ import annotations
from typing import Optional, Callable, Dict
import torch
import torch.nn as nn
from torch.distributions import Normal

from .feature_extractors.state_encoder import *
from rsl_rl.utils import resolve_nn_activation

# Simple registry to avoid eval-based class lookup.
ACTOR_REGISTRY: Dict[str, Callable] = {}


def register_actor(name: str):
    def decorator(cls):
        ACTOR_REGISTRY[name] = cls
        return cls
    return decorator


@register_actor("Actor")
class Actor(nn.Module):
    def __init__(self, 
                 num_actions, 
                 scan_encoder_dims,
                 actor_hidden_dims, 
                 priv_encoder_dims, 
                 activation, 
                 tanh_encoder_output=False,
                 **kwargs
                 ) -> None:
        super().__init__()
        # prop -> scan -> priv_explicit -> priv_latent -> hist
        # actor input: prop -> scan -> priv_explicit -> latent
        self.num_prop = num_prop = kwargs.pop('num_prop')
        self.num_scan = num_scan = kwargs.pop('num_scan')
        self.num_hist = num_hist = kwargs.pop('num_hist')
        self.num_actions = num_actions
        self.num_priv_latent = num_priv_latent = kwargs.pop('num_priv_latent')
        self.num_priv_explicit = num_priv_explicit = kwargs.pop('num_priv_explicit')
        self.if_scan_encode = scan_encoder_dims is not None and num_scan > 0
        self.in_features = num_prop + num_scan + num_priv_latent + num_priv_explicit + num_prop * num_hist ##  +  

        if len(priv_encoder_dims) > 0:
            priv_encoder_layers = []
            priv_encoder_layers.append(nn.Linear(num_priv_latent, priv_encoder_dims[0]))
            priv_encoder_layers.append(activation)
            for l in range(len(priv_encoder_dims) - 1):
                priv_encoder_layers.append(nn.Linear(priv_encoder_dims[l], priv_encoder_dims[l + 1]))
                priv_encoder_layers.append(activation)
            self.priv_encoder = nn.Sequential(*priv_encoder_layers)
            priv_encoder_output_dim = priv_encoder_dims[-1]
        else:
            self.priv_encoder = nn.Identity()
            priv_encoder_output_dim = num_priv_latent

        state_history_encoder_cfg = kwargs.pop('state_history_encoder')
        state_histroy_encoder_class = eval(state_history_encoder_cfg.pop('class_name'))
        self.history_encoder: StateHistoryEncoder = state_histroy_encoder_class(
                                                            activation, 
                                                            num_prop, 
                                                            num_hist, 
                                                            priv_encoder_output_dim,
                                                            state_history_encoder_cfg.pop('channel_size')
                                                            )
        if self.if_scan_encode:
            scan_encoder = []
            scan_encoder.append(nn.Linear(num_scan, scan_encoder_dims[0]))
            scan_encoder.append(activation)
            for l in range(len(scan_encoder_dims) - 1):
                if l == len(scan_encoder_dims) - 2:
                    scan_encoder.append(nn.Linear(scan_encoder_dims[l], scan_encoder_dims[l+1]))
                    scan_encoder.append(nn.Tanh())
                else:
                    scan_encoder.append(nn.Linear(scan_encoder_dims[l], scan_encoder_dims[l + 1]))
                    scan_encoder.append(activation)
            self.scan_encoder = nn.Sequential(*scan_encoder)
            self.scan_encoder_output_dim = scan_encoder_dims[-1]
        else:
            self.scan_encoder = nn.Identity()
            self.scan_encoder_output_dim = num_scan
        
        actor_layers = []
        actor_layers.append(nn.Linear(num_prop+
                                      self.scan_encoder_output_dim+
                                      num_priv_explicit+
                                      priv_encoder_output_dim, 
                                      actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        if tanh_encoder_output:
            actor_layers.append(nn.Tanh())
        self.actor_backbone = nn.Sequential(*actor_layers)

    def forward(
        self, 
        obs, 
        hist_encoding: bool, 
        scandots_latent: Optional[torch.Tensor] = None
        ):
        if self.if_scan_encode:
            obs_scan = obs[:, self.num_prop:self.num_prop + self.num_scan]
            if scandots_latent is None:
                scan_latent = self.scan_encoder(obs_scan)   
            else:
                scan_latent = scandots_latent
            obs_prop_scan = torch.cat([obs[:, :self.num_prop], scan_latent], dim=1)
        else:
            obs_prop_scan = obs[:, :self.num_prop + self.num_scan]
        obs_priv_explicit = obs[:, self.num_prop + self.num_scan:self.num_prop + self.num_scan + self.num_priv_explicit]
        if hist_encoding:
            latent = self.infer_hist_latent(obs)
        else:
            latent = self.infer_priv_latent(obs)
        backbone_input = torch.cat([obs_prop_scan, obs_priv_explicit, latent], dim=1)
        backbone_output = self.actor_backbone(backbone_input)
        return backbone_output
    
    def infer_priv_latent(self, obs):
        priv = obs[:, self.num_prop + self.num_scan + self.num_priv_explicit: self.num_prop + self.num_scan + self.num_priv_explicit + self.num_priv_latent]
        return self.priv_encoder(priv)
    
    def infer_hist_latent(self, obs):
        hist = obs[:, -self.num_hist*self.num_prop:]
        return self.history_encoder(hist.view(-1, self.num_hist, self.num_prop))
    
    def infer_scandots_latent(self, obs):
        scan = obs[:, self.num_prop:self.num_prop + self.num_scan]
        return self.scan_encoder(scan)


@register_actor("GatedDualHeadActor")
class GatedDualHeadActor(nn.Module):
    """Two expert heads (jump / crawl) mixed by a gating MLP using hurdle features."""
    def __init__(
        self,
        num_actions,
        scan_encoder_dims,
        actor_hidden_dims,
        priv_encoder_dims,
        activation,
        tanh_encoder_output=False,
        gating_hidden_dims=None,
        gating_temperature: float = 1.0,
        gating_input_indices=None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.num_prop = num_prop = kwargs.pop("num_prop")
        self.num_scan = num_scan = kwargs.pop("num_scan")
        self.num_hist = num_hist = kwargs.pop("num_hist")
        self.num_actions = num_actions
        self.num_priv_latent = num_priv_latent = kwargs.pop("num_priv_latent")
        self.num_priv_explicit = num_priv_explicit = kwargs.pop("num_priv_explicit")
        self.if_scan_encode = scan_encoder_dims is not None and num_scan > 0
        self.in_features = num_prop + num_scan + num_priv_latent + num_priv_explicit + num_prop * num_hist

        if len(priv_encoder_dims) > 0:
            priv_encoder_layers = [nn.Linear(num_priv_latent, priv_encoder_dims[0]), activation]
            for l in range(len(priv_encoder_dims) - 1):
                priv_encoder_layers.append(nn.Linear(priv_encoder_dims[l], priv_encoder_dims[l + 1]))
                priv_encoder_layers.append(activation)
            self.priv_encoder = nn.Sequential(*priv_encoder_layers)
            priv_encoder_output_dim = priv_encoder_dims[-1]
        else:
            self.priv_encoder = nn.Identity()
            priv_encoder_output_dim = num_priv_latent

        state_history_encoder_cfg = kwargs.pop("state_history_encoder")
        state_histroy_encoder_class = eval(state_history_encoder_cfg.pop("class_name"))
        self.history_encoder: StateHistoryEncoder = state_histroy_encoder_class(
            activation,
            num_prop,
            num_hist,
            priv_encoder_output_dim,
            state_history_encoder_cfg.pop("channel_size"),
        )
        if self.if_scan_encode:
            scan_encoder = [nn.Linear(num_scan, scan_encoder_dims[0]), activation]
            for l in range(len(scan_encoder_dims) - 1):
                if l == len(scan_encoder_dims) - 2:
                    scan_encoder.append(nn.Linear(scan_encoder_dims[l], scan_encoder_dims[l + 1]))
                    scan_encoder.append(nn.Tanh())
                else:
                    scan_encoder.append(nn.Linear(scan_encoder_dims[l], scan_encoder_dims[l + 1]))
                    scan_encoder.append(activation)
            self.scan_encoder = nn.Sequential(*scan_encoder)
            self.scan_encoder_output_dim = scan_encoder_dims[-1]
        else:
            self.scan_encoder = nn.Identity()
            self.scan_encoder_output_dim = num_scan

        backbone_in_dim = (
            num_prop + self.scan_encoder_output_dim + num_priv_explicit + priv_encoder_output_dim
        )
        def build_head():
            layers = [nn.Linear(backbone_in_dim, actor_hidden_dims[0]), activation]
            for l in range(len(actor_hidden_dims)):
                if l == len(actor_hidden_dims) - 1:
                    layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
                else:
                    layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                    layers.append(activation)
            if tanh_encoder_output:
                layers.append(nn.Tanh())
            return nn.Sequential(*layers)

        self.jump_head = build_head()
        self.crawl_head = build_head()

        gating_hidden_dims = gating_hidden_dims or [64, 64]
        gate_layers = []
        self.gating_input_indices = gating_input_indices or list(range(min(8, num_priv_explicit)))
        gate_in = len(self.gating_input_indices)
        gate_layers.append(nn.Linear(gate_in, gating_hidden_dims[0]))
        gate_layers.append(activation)
        for l in range(len(gating_hidden_dims) - 1):
            gate_layers.append(nn.Linear(gating_hidden_dims[l], gating_hidden_dims[l + 1]))
            gate_layers.append(activation)
        gate_layers.append(nn.Linear(gating_hidden_dims[-1], 2))
        self.gating_mlp = nn.Sequential(*gate_layers)
        self.gating_temperature = gating_temperature

    def _split_inputs(self, obs, scandots_latent=None):
        if self.if_scan_encode:
            obs_scan = obs[:, self.num_prop : self.num_prop + self.num_scan]
            scan_latent = self.scan_encoder(obs_scan) if scandots_latent is None else scandots_latent
            obs_prop_scan = torch.cat([obs[:, : self.num_prop], scan_latent], dim=1)
        else:
            obs_prop_scan = obs[:, : self.num_prop + self.num_scan]
        obs_priv_explicit = obs[:, self.num_prop + self.num_scan : self.num_prop + self.num_scan + self.num_priv_explicit]
        return obs_prop_scan, obs_priv_explicit

    def _latent(self, obs, hist_encoding):
        if hist_encoding:
            hist = obs[:, -self.num_hist * self.num_prop :]
            return self.history_encoder(hist.view(-1, self.num_hist, self.num_prop))
        priv = obs[:, self.num_prop + self.num_scan + self.num_priv_explicit : self.num_prop + self.num_scan + self.num_priv_explicit + self.num_priv_latent]
        return self.priv_encoder(priv)

    def forward(self, obs, hist_encoding: bool, scandots_latent: Optional[torch.Tensor] = None):
        obs_prop_scan, obs_priv_explicit = self._split_inputs(obs, scandots_latent)
        latent = self._latent(obs, hist_encoding)
        backbone_input = torch.cat([obs_prop_scan, obs_priv_explicit, latent], dim=1)

        a_jump = self.jump_head(backbone_input)
        a_crawl = self.crawl_head(backbone_input)

        gate_feat = obs_priv_explicit[:, self.gating_input_indices]
        gate_logits = self.gating_mlp(gate_feat) / max(self.gating_temperature, 1e-3)
        gate = torch.softmax(gate_logits, dim=-1)
        # expose gate statistics for logging
        self.gate_last = gate.detach()
        self.gate_entropy = (-(gate * torch.log(gate + 1e-8)).sum(dim=-1)).detach()
        mixed = gate[:, 0:1] * a_jump + gate[:, 1:2] * a_crawl
        return mixed

class ActorCriticRMA(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        super(ActorCriticRMA, self).__init__()

        self.kwargs = kwargs
        priv_encoder_dims = kwargs['priv_encoder_dims']
        scan_encoder_dims = kwargs['scan_encoder_dims']
        activation = resolve_nn_activation(activation)
        actor_cfg = kwargs.pop('actor')
        actor_class_name = actor_cfg.pop('class_name')
        if actor_class_name not in ACTOR_REGISTRY:
            raise KeyError(f"Actor class '{actor_class_name}' not registered.")
        actor_class = ACTOR_REGISTRY[actor_class_name]
        self.actor: Actor = actor_class(
                                num_actions, 
                                scan_encoder_dims,
                                actor_hidden_dims, 
                                priv_encoder_dims, 
                                activation, 
                                tanh_encoder_output=kwargs['tanh_encoder_output'],
                                **actor_cfg
                                )
        
        mlp_input_dim_c = num_critic_obs
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        self.distribution = None
        Normal.set_default_validate_args = False

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations, hist_encoding):
        mean = self.actor(observations, hist_encoding)
        if self.noise_std_type == "scalar":
            std = mean*0. + self.std
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        self.distribution = Normal(mean, std)

    def act(self, observations, hist_encoding=False, **kwargs):
        self.update_distribution(observations, hist_encoding)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)
    
    def act_inference(self, observations, hist_encoding=False, scandots_latent=None, **kwargs):
        actions_mean = self.actor(observations, hist_encoding, scandots_latent)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        return True
