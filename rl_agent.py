"""
Multi-Objective Parameter-Exploring Policy Gradient (PEPG) Agent
================================================================
This module implements the reinforcement learning agent for active quantum error
correction fine-tuning and drift tracking, following the Google Quantum AI
methodology (arXiv:2511.08493).

Key Features:
1. Factorized continuous Gaussian policy over physical control parameters: N(mu, Sigma)
2. Local gradient masking via the bipartite factor-graph matrix M (O x P)
3. Multi-objective advantage formulation with per-detector baseline tracking
4. Entropy regularization for sustained exploration in non-stationary environments
5. Antithetic parameter sampling for variance reduction
"""

from typing import Dict, Optional, Tuple
import numpy as np


class RLControlAgent:
    """
    Reinforcement learning agent that continuously steers physical control parameters
    from QEC error detection rates (EDR).
    """

    def __init__(
        self,
        num_params: int,
        num_detectors: int,
        mask_matrix: np.ndarray,
        lr_mu: float = 0.04,
        lr_sigma: float = 0.015,
        lr_baseline: float = 0.1,
        entropy_reg: float = 0.01,
        sigma_init: float = 0.2,
        sigma_min: float = 0.05,
        sigma_max: float = 0.4,
        grad_clip: float = 2.0,
        use_adam: bool = True,
        seed: int = 42,
    ):
        self.num_params = num_params
        self.num_detectors = num_detectors
        self.M = mask_matrix.astype(np.float32)  # Shape (O, P)
        self.lr_mu = lr_mu
        self.lr_sigma = lr_sigma
        self.lr_baseline = lr_baseline
        self.entropy_reg = entropy_reg
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.grad_clip = grad_clip
        self.use_adam = use_adam
        self.rng = np.random.default_rng(seed)

        # Policy parameters: mean mu and std sigma
        self.mu = np.zeros(num_params, dtype=np.float32)
        self.sigma = np.full(num_params, sigma_init, dtype=np.float32)

        # Local advantage baseline vector b in R^O
        self.baseline = np.zeros(num_detectors, dtype=np.float32)

        # Precompute column sums of M for normalized masked advantage
        self.M_col_sums = np.sum(self.M, axis=0, keepdims=True)  # Shape (1, P)
        self.M_col_sums = np.maximum(self.M_col_sums, 1.0)

        # Adam optimizer state
        if self.use_adam:
            self.m_mu = np.zeros(num_params, dtype=np.float32)
            self.v_mu = np.zeros(num_params, dtype=np.float32)
            self.m_sig = np.zeros(num_params, dtype=np.float32)
            self.v_sig = np.zeros(num_params, dtype=np.float32)
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.eps_adam = 1e-8
            self.step_count = 0

    def sample_policies(self, batch_size: int, antithetic: bool = True) -> np.ndarray:
        """
        Samples candidate policy parameter vectors p ~ N(mu, Sigma).
        Uses antithetic perturbation pairs (mu + delta, mu - delta) if antithetic=True.
        Returns:
            np.ndarray of shape (batch_size, num_params)
        """
        if antithetic:
            half_batch = (batch_size + 1) // 2
            noise = self.rng.normal(0.0, 1.0, size=(half_batch, self.num_params)).astype(np.float32)
            delta = noise * self.sigma
            pos = self.mu + delta
            neg = self.mu - delta
            samples = np.vstack([pos, neg])[:batch_size]
        else:
            noise = self.rng.normal(0.0, 1.0, size=(batch_size, self.num_params)).astype(np.float32)
            samples = self.mu + noise * self.sigma

        return samples

    def update(
        self,
        actions: np.ndarray,
        observations: np.ndarray,
    ) -> Dict[str, float]:
        """
        Performs masked policy gradient update given a batch of actions and observations.

        Args:
            actions: sampled parameter vectors, shape (B, P)
            observations: empirical detector error rates o_k in [0, 1]^O, shape (B, O)

        Returns:
            dict of training diagnostics (mean_reward, norm_grad_mu, mean_sigma, etc.)
        """
        B = actions.shape[0]

        # 1. Multi-objective rewards: negative detector rates r = -o
        rewards = -observations  # Shape (B, O)
        mean_reward = float(np.mean(rewards))

        # 2. Local advantage vectors: alpha_k = r_k - b
        adv_per_detector = rewards - self.baseline  # Shape (B, O)

        # 3. Masked parameter advantages using the bipartite graph M:
        # A_{k, i} = sum_j M_{ji} * alpha_{k, j} / sum_j M_{ji}
        masked_advantage = (adv_per_detector @ self.M) / self.M_col_sums  # Shape (B, P)

        # 4. Analytic policy gradients for factorized Gaussian:
        # grad_mu = E[ A_i * (p_i - mu_i) / sigma_i^2 ]
        diff = actions - self.mu  # Shape (B, P)
        grad_mu = np.mean(masked_advantage * diff / (self.sigma ** 2), axis=0)

        # grad_sigma = E[ A_i * ((p_i - mu_i)^2 - sigma_i^2) / sigma_i^3 ] + entropy_reg / sigma_i
        grad_sigma = (
            np.mean(masked_advantage * (diff ** 2 - self.sigma ** 2) / (self.sigma ** 3), axis=0)
            + (self.entropy_reg / self.sigma)
        )

        # Gradient clipping for stability
        if self.grad_clip > 0:
            norm_mu = np.linalg.norm(grad_mu)
            if norm_mu > self.grad_clip:
                grad_mu = grad_mu * (self.grad_clip / norm_mu)
            norm_sig = np.linalg.norm(grad_sigma)
            if norm_sig > self.grad_clip:
                grad_sigma = grad_sigma * (self.grad_clip / norm_sig)

        # 5. Parameter updates
        if self.use_adam:
            self.step_count += 1
            # Adam for mu
            self.m_mu = self.beta1 * self.m_mu + (1 - self.beta1) * grad_mu
            self.v_mu = self.beta2 * self.v_mu + (1 - self.beta2) * (grad_mu ** 2)
            m_hat_mu = self.m_mu / (1 - self.beta1 ** self.step_count)
            v_hat_mu = self.v_mu / (1 - self.beta2 ** self.step_count)
            self.mu += self.lr_mu * m_hat_mu / (np.sqrt(v_hat_mu) + self.eps_adam)

            # Adam for sigma
            self.m_sig = self.beta1 * self.m_sig + (1 - self.beta1) * grad_sigma
            self.v_sig = self.beta2 * self.v_sig + (1 - self.beta2) * (grad_sigma ** 2)
            m_hat_sig = self.m_sig / (1 - self.beta1 ** self.step_count)
            v_hat_sig = self.v_sig / (1 - self.beta2 ** self.step_count)
            self.sigma += self.lr_sigma * m_hat_sig / (np.sqrt(v_hat_sig) + self.eps_adam)
        else:
            self.mu += self.lr_mu * grad_mu
            self.sigma += self.lr_sigma * grad_sigma

        # Enforce exploration variance bounds
        self.sigma = np.clip(self.sigma, self.sigma_min, self.sigma_max)

        # 6. Update baseline: smooth exponential moving average
        batch_mean_reward = np.mean(rewards, axis=0)
        self.baseline = (1 - self.lr_baseline) * self.baseline + self.lr_baseline * batch_mean_reward

        return {
            "mean_reward": mean_reward,
            "norm_grad_mu": float(np.linalg.norm(grad_mu)),
            "mean_sigma": float(np.mean(self.sigma)),
            "mean_baseline": float(np.mean(self.baseline)),
        }

    def get_policy_stats(self) -> Dict[str, np.ndarray]:
        """Returns the current policy mean and standard deviation."""
        return {
            "mu": self.mu.copy(),
            "sigma": self.sigma.copy(),
        }
