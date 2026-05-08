r"""
src/actor_critic.py

PPO + RV network using shared CNN encoder, policy head, and Siamese Relative Critic.

References:
    - Architecture overview:                                    Sec. 5.1
    - Relative critic head:                                     Eq. 30
    - Antisymmetry and zero self-difference guarantee:          Eq. 30
    - Orthogonal weight initialization:                         Sec. 6
    - Non-linear antisymmetric MLPs (tanh):                     Sec. 5.1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Tuple


class AtariEncoder(nn.Module):
    r"""
    PPO CNN encoder for Atari/Minatar.
    """

    def __init__(self, in_channels: int = 4, hidden_dim: int = 512, input_size: int = 84):

        super().__init__()
        self.cnn_out_dim = 64 * 7 * 7
        self.hidden_dim = hidden_dim
        self.in_channels = in_channels

        # minatar size
        if input_size <= 16:
            self.cnn = nn.Sequential(
                nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Flatten(),
            )

        # atari size
        else:
            self.cnn = nn.Sequential( # (32, 20, 20) -> (64, 9, 9) -> (64, 7, 7)
                nn.Conv2d(self.in_channels, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
            )

        # self.fc = nn.Sequential(
        #     nn.Linear(self.cnn_out_dim, self.hidden_dim),
        #     nn.ReLU(),
        # )

        # infer CNN output dimension dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, input_size, input_size)
            cnn_out_dim = self.cnn(dummy).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(cnn_out_dim, hidden_dim),
            nn.ReLU(),
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            x:          input of shape (B, C, H, W).

        Returns:
            embedding:  embeddings of shape (B, hidden_dim).
        """

        return self.fc(self.cnn(x))


class RelativeCritic(nn.Module):
    r"""
    Siamese relative critic head \phi(f_enc(s_i) - f_enc(s_j)) using a 
    single learned vector w \in R^d. This ensures zero self-difference 
    and antisymmetry.

    Note that non-linear antisymmetric MLPs (tanh activations) match constraints 
    but did not improve results.
    """

    # TODO implement non-linear MLPs

    def __init__(self, hidden_dim: int = 512):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.w = nn.Linear(self.hidden_dim, 1, bias=False)

    def forward(self, emb_i: torch.Tensor, emb_j: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            emb_i:                      f_enc(s_i) input of shape (B, d).
        emb_j:                          f_enc(s_j) input of shape (B, d).

        Returns:
            \Delta_\theta(s_i, s_j):    w^T(emb_i - emb_j) of shape (B,).
        """
        return self.w(emb_i - emb_j).squeeze(-1)


class PPORVActorCritic(nn.Module):
    r"""
    Full PPO + RV model, shared encoder -> policy head + relative critic head.
    The encoder is shared between actor and critic. The relative critic uses a 
    siamese forward pass which encodes both states with the same encoder, then 
    projects the embedding difference.

    The weights are initialized orthogonally with a small policy logit scale.
    """

    def __init__(self, n_actions: int, in_channels: int = 4, hidden_dim: int = 512, input_size: int = 84, policy_logit_scale: float = 0.01):
        super().__init__()
        self.policy_logit_scale = policy_logit_scale
        
        self.encoder = AtariEncoder(
            in_channels, 
            hidden_dim,
            input_size
        )
        self.policy_head = nn.Linear(hidden_dim, n_actions)
        self.critic_head = RelativeCritic(hidden_dim)
        self._init_weights(self.policy_logit_scale)

    def _init_weights(self, policy_logit_scale: float):
        r"""
        Orthogonal initialization of weights.
        """
        
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=1.0)

                # zero out bias
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        nn.init.orthogonal_(self.policy_head.weight, gain=policy_logit_scale)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        r"""
        Encode a batch of observations. 

        Args:
            obs:        input of shape (B, C, H, W).
        Returns:
            f_enc(obs): encoded output
        """

        return self.encoder(obs)

    def policy(self, obs: torch.Tensor) -> Categorical:
        r"""
        Return action distribution for a batch of observations.

        Args:
            obs:        input of shape (B, C, H, W).
        Returns:
            actions:    a Categorical distribution of actions.
        """

        emb = self.encode(obs)
        logits = self.policy_head(emb)
        return Categorical(logits=logits)

    def delta(self, obs_i: torch.Tensor, obs_j: torch.Tensor) -> torch.Tensor:
        r"""
        Compute pairwise value difference \Delta_\theta(s_i, s_j).

        Encodes both states using shared encoder and projects the
        embedding difference.

        Args:
            obs_i:  input observation tensor of shape (B, C, H, W).
            obs_j:  input observation tensor of shape (B, C, H, W).

        Returns:
            delta:  antisymmetric value difference of shape (B,).
        """

        emb_i = self.encode(obs_i)
        emb_j = self.encode(obs_j)
        return self.critic_head(emb_i, emb_j)

    def forward(
        self,
        obs_i: torch.Tensor,
        obs_j: torch.Tensor,
    ) -> Tuple[Categorical, torch.Tensor]:
        r"""
        Joint forward pass for training loop. The action distribution is 
        used for the policy gradient. The delta value is used for the
        critic loss.
        
        Note that obs_i is the "reference" state for the policy. The critic
        is trained on arbitrary pairs (i, j) sampled from the rollout buffer.

        Args:
            obs_i:  input observation tensor of shape (B, C, H, W).
            obs_j:  input observation tensor of shape (B, C, H, W).

        Returns:
            dist:   action distribution for obs_i.
            delta:  \Delta_\theta(obs_i, obs_j).
        """

        emb_i  = self.encode(obs_i)
        emb_j  = self.encode(obs_j)

        logits = self.policy_head(emb_i)
        dist   = Categorical(logits=logits)
        delta  = self.critic_head(emb_i, emb_j)

        return dist, delta