import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from math import sqrt


def combine_latent_predictions(mu_predicted, cov_predicted, mu_q_phi, cov_q_phi):
    # z_predicted: Predicted distribution from transition model, e.g., Gaussian parameters (mean, covariance)
    # q_phi_z_next: Posterior distribution from recognition network, e.g., Gaussian parameters (mean, covariance)

    # Perform Bayesian update to combine latent predictions
    z_mean = (
        (cov_predicted @ torch.linalg.inv(cov_predicted + cov_q_phi))
        @ mu_q_phi.unsqueeze(-1)
        + (cov_q_phi @ torch.linalg.inv(cov_predicted + cov_q_phi))
        @ mu_predicted.unsqueeze(-1)
    ).squeeze(-1)

    P_z = cov_predicted @ torch.linalg.inv(cov_predicted + cov_q_phi)

    # Additional preprocessing needed for covariance to be p.s.d and symmetric
    scale_term = 1.1 * torch.ones([P_z.shape[0], 6], device=P_z.device)
    P_z = 0.5 * (P_z + P_z.transpose(-2, -1))
    min_eigen = torch.min(torch.linalg.eigvals(P_z).to(torch.float32))
    if min_eigen < 0:
        P_z -= min_eigen * torch.diag_embed(scale_term)

    return MultivariateNormal(loc=z_mean, covariance_matrix=P_z)


# =======================
# Covariance Conversion (still used for P_pred and R_pred)
# =======================
def convert2cov(mat, state_dim=5, scale_val=2.0):
    scale_term = scale_val * torch.ones([mat.shape[0], state_dim], device=mat.device)
    mat = 0.5 * (mat + mat.transpose(-2, -1))
    min_eigen = torch.min(torch.linalg.eigvals(mat).real)
    if min_eigen < 0:
        mat -= min_eigen * torch.diag_embed(scale_term)
    return mat


# =======================
# Residual Self-Attention Block
# =======================
class FeatureSelfAttention(nn.Module):
    """
    Self-attention over feature dimension.
    Input:  x in R^{B, F, D}
    Output: same shape, but each feature attends to all others.
    """
    def __init__(self, feature_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x):
        # x: [B, F, D]
        out, _ = self.attn(x, x, x)
        return out


class ResidualSelfAttentionEncoder(nn.Module):
    """
    Drop-in replacement for TransformerBlock:
    - Treats input_dim as "sequence length" (features)
    - Projects scalar features to feature_dim
    - Applies feature-wise self-attention + LayerNorm
    - Flattens and passes through residual MLP stack
    - Outputs [B, hidden_dim]
    """
    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        feature_dim=64,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
        residual_scale=0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.residual_scale = residual_scale

        # Project each scalar feature -> feature_dim
        self.input_proj = nn.Linear(1, feature_dim)

        # Self-attention over features
        self.attention = FeatureSelfAttention(feature_dim, num_heads=num_heads, dropout=dropout)
        self.attn_norm = nn.LayerNorm(feature_dim)

        # Flatten feature embeddings -> hidden_dim
        self.flatten_proj = nn.Linear(input_dim * feature_dim, hidden_dim)

        # Residual MLP stack
        self.mlp_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        """
        x: [B, input_dim]
        returns: [B, hidden_dim]
        """
        B = x.size(0)

        # [B, F] -> [B, F, 1]
        x = x.unsqueeze(-1)
        # [B, F, 1] -> [B, F, feature_dim]
        x = self.input_proj(x)

        # Attention over features
        x = self.attention(x)
        x = self.attn_norm(x)

        # Flatten
        x = x.reshape(B, -1)  # [B, F * feature_dim]
        x = self.flatten_proj(x)  # [B, hidden_dim]

        # Residual MLP
        for layer in self.mlp_layers:
            residual = x
            x = layer(x)
            x = x + self.residual_scale * residual

        return x


# =======================
# DKF Component Modules
# =======================
class ProcessNoiseNet(nn.Module):
    """
    Predicts process noise covariance Q_k using low-rank approximation:
      Q_k = diag(sigma_diag) + U U^T
    """
    def __init__(self, latent_size, hidden_size, control_size,
                 feature_dim=64, num_layers=2, num_heads=4, dropout=0.1,
                 rank=1):
        super().__init__()
        input_dim = latent_size + control_size + 1
        self.encoder = ResidualSelfAttentionEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_size,
            feature_dim=feature_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.latent_size = latent_size
        self.rank = rank

        # Output: diag (latent_size) + low-rank vec (latent_size * rank)
        self.fc_out = nn.Linear(hidden_size, latent_size + latent_size * rank)

    def forward(self, z_prev, u, dt):
        x = torch.cat((z_prev, u, dt), dim=1)  # [B, latent+control+1]
        x = self.encoder(x)

        out = self.fc_out(x)  # [B, latent_size + latent_size*rank]
        B = out.size(0)
        d = self.latent_size

        diag_raw = out[:, :d]
        low_rank_raw = out[:, d:]  # [B, d * rank]

        # Ensure positive diagonal
        sigma_diag = F.softplus(diag_raw) + 1e-4  # [B, d]

        # Low-rank factor U
        U = low_rank_raw.view(B, d, self.rank)  # [B, d, r]

        # Q_k = diag(sigma_diag) + U U^T
        Q_k = torch.diag_embed(sigma_diag) + torch.matmul(U, U.transpose(-1, -2))

        return Q_k


class MeasurementNoiseNet(nn.Module):
    """
    Predicts measurement noise covariance R_k using low-rank approximation:
      R_k = diag(sigma_diag) + U U^T
    """
    def __init__(self, latent_size, hidden_size, observed_size,
                 feature_dim=64, num_layers=2, num_heads=4, dropout=0.1,
                 rank=1):
        super().__init__()
        input_dim = latent_size
        self.encoder = ResidualSelfAttentionEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_size,
            feature_dim=feature_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.observed_size = observed_size
        self.rank = rank

        # Output: diag (obs_size) + low-rank vec (obs_size * rank)
        self.fc_out = nn.Linear(hidden_size, observed_size + observed_size * rank)

    def forward(self, z_next):
        x = self.encoder(z_next)
        out = self.fc_out(x)  # [B, obs + obs*rank]
        B = out.size(0)
        d = self.observed_size

        diag_raw = out[:, :d]
        low_rank_raw = out[:, d:]  # [B, d * rank]

        sigma_diag = F.softplus(diag_raw) + 1e-4  # [B, d]
        U = low_rank_raw.view(B, d, self.rank)    # [B, d, r]

        R_k = torch.diag_embed(sigma_diag) + torch.matmul(U, U.transpose(-1, -2))

        return R_k


class DirectTransitionNet(nn.Module):
    def __init__(self, latent_size, control_size, hidden_size,
                 feature_dim=64, num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()
        input_dim = latent_size + control_size + 1
        self.encoder = ResidualSelfAttentionEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_size,
            feature_dim=feature_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.fc_out = nn.Linear(hidden_size, 2 * latent_size)
        self.latent_size = latent_size

    def forward(self, prev_z, u, dt, Q_k):
        x = torch.cat((prev_z, u, dt), dim=1)
        x = self.encoder(x)
        output = self.fc_out(x)
        z_pred = output[:, :self.latent_size]
        log_sigma = output[:, self.latent_size:]
        P_pred = 0.9 * F.softplus(torch.diag_embed(log_sigma) + Q_k) + 0.1
        return MultivariateNormal(
            loc=z_pred,
            covariance_matrix=convert2cov(P_pred, state_dim=self.latent_size),
        )


class DirectMeasurementNet(nn.Module):
    def __init__(self, latent_size, hidden_size, observed_size,
                 feature_dim=64, num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()
        input_dim = latent_size
        self.encoder = ResidualSelfAttentionEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_size,
            feature_dim=feature_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.fc_out = nn.Linear(hidden_size, 2 * observed_size)
        self.observed_size = observed_size

    def forward(self, z_k, R_k):
        x = self.encoder(z_k)
        output = self.fc_out(x)
        x_pred = output[:, :self.observed_size]
        log_sigma = output[:, self.observed_size:]
        R_pred = 0.9 * F.softplus(torch.diag_embed(log_sigma) + R_k) + 0.1
        return MultivariateNormal(
            loc=x_pred,
            covariance_matrix=convert2cov(R_pred, state_dim=self.observed_size),
        )


# =======================
# Deep Kalman Filter
# =======================
class DeepKalmanFilter(nn.Module):
    def __init__(
        self,
        hidden_size=128,
        observed_size=5,
        latent_size=5,
        control_size=2,
        batch_size=1000,
        feature_dim=64,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
        rank_Q=1,
        rank_R=1,
    ):
        super().__init__()
        self.transition_net = DirectTransitionNet(
            latent_size, control_size, hidden_size,
            feature_dim=feature_dim, num_layers=num_layers,
            num_heads=num_heads, dropout=dropout,
        )
        self.measurement_net = DirectMeasurementNet(
            latent_size, hidden_size, observed_size,
            feature_dim=feature_dim, num_layers=num_layers,
            num_heads=num_heads, dropout=dropout,
        )
        self.compute_Q_k = ProcessNoiseNet(
            latent_size, hidden_size, control_size,
            feature_dim=feature_dim, num_layers=num_layers,
            num_heads=num_heads, dropout=dropout,
            rank=rank_Q,
        )
        self.compute_R_k = MeasurementNoiseNet(
            latent_size, hidden_size, observed_size,
            feature_dim=feature_dim, num_layers=num_layers,
            num_heads=num_heads, dropout=dropout,
            rank=rank_R,
        )
        self.latent_size = latent_size
        self.batch_size = batch_size
        print("DeepKalmanFilter with residual self-attention + low-rank Q/R initialized")

    def jacobian_observation_model(self, z, R_k, no_grad=False):
        dist_x_next = self.measurement_net(z, R_k)
        x_pred = dist_x_next.loc
        grad_dims = torch.ones_like(x_pred)
        if no_grad:
            with torch.no_grad():
                jacobian = torch.autograd.grad(
                    outputs=x_pred,
                    inputs=z,
                    grad_outputs=grad_dims,
                    retain_graph=True,
                )[0]
        else:
            jacobian = torch.autograd.grad(
                outputs=x_pred,
                inputs=z,
                grad_outputs=grad_dims,
                retain_graph=True,
                )[0]
        return jacobian

    def forward(self, latent_transitions, observed_transitions, is_testing=False):
        # Expect latent_transitions: [..., >=8], observed_transitions: [..., >=?]
        z_prev = latent_transitions[:, :5]
        x_next = observed_transitions[:, 8:]
        uu = latent_transitions[:, 6:8]
        dt = latent_transitions[:, 5:6]

        # Prediction step
        Q_k = self.compute_Q_k(z_prev, uu, dt)
        next_z_prior = self.transition_net(z_prev, uu, dt, Q_k)
        z_pred = next_z_prior.loc
        P_z = next_z_prior.covariance_matrix

        # Measurement prediction
        R_k = self.compute_R_k(z_pred)
        dist_x_pred = self.measurement_net(z_pred, R_k)
        x_pred = dist_x_pred.loc
        R_x = dist_x_pred.covariance_matrix

        # Linearization (Jacobian)
        if is_testing:
            C_k = torch.diag_embed(self.jacobian_observation_model(z_pred, R_k, no_grad=True))
        else:
            C_k = torch.diag_embed(self.jacobian_observation_model(z_pred, R_k))

        # Kalman update
        innovation = x_next - x_pred
        S = torch.matmul(R_x, R_x.transpose(-2, -1)) + R_k
        K = torch.matmul(P_z, torch.matmul(C_k, torch.inverse(S)))
        z_correct = torch.matmul(K, innovation.unsqueeze(-1)).squeeze(-1)
        z_next = z_pred + z_correct
        P_z = P_z - torch.matmul(K, torch.matmul(C_k, P_z))

        return z_next, P_z, x_pred, R_x
