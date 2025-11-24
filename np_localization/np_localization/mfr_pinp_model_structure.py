import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent, kl_divergence

class MLP(nn.Module):

  def __init__(
      self,
      input_size,
      output_size,
      hidden_size=128,
      n_hidden_layers=1,
      activation=nn.ReLU(),
      is_bias=True,
      dropout=0,
      aggregate_step=False,
      num_latents=0,
      device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):

    super(MLP, self).__init__()
    self.input_size = input_size
    self.output_size = output_size
    self.hidden_size = hidden_size
    self.n_hidden_layers = n_hidden_layers

    self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
    self.activation = activation

    self.to_hidden = nn.Linear(self.input_size, self.hidden_size, bias=is_bias).to(device)
    self.linears = nn.ModuleList(
        [
            nn.Linear(self.hidden_size, self.hidden_size, bias=is_bias)
            for _ in range(self.n_hidden_layers - 1)
        ]
    ).to(device)

    self.out = nn.Linear(self.hidden_size, self.output_size, bias=is_bias).to(device)
    self.aggregate_step = aggregate_step
    if self.aggregate_step:
      self.num_latents = num_latents
      self.latent_hidden = int((hidden_size+num_latents)/2)
      self.penultimate_layer = nn.Linear(output_size, self.latent_hidden, bias=is_bias).to(device)
      self.mu_layer = nn.Linear(self.latent_hidden, num_latents).to(device)
      self.log_sigma_layer = nn.Linear(self.latent_hidden, num_latents).to(device)


  def forward(self, x):
    # Ensure input tensor is on the same device as the model
    x = x.to(self.to_hidden.weight.device)

    out = self.to_hidden(x)
    out = self.activation(out)
    x = self.dropout(out)

    for linear in self.linears:
      out = linear(x)
      out = self.activation(out)
      out = self.dropout(out)
      x = out

    out = self.out(out)

    # Add-in for latent encoder steps
    if self.aggregate_step:
      out = torch.mean(out, dim=1)
      out = self.penultimate_layer(out)
      mu = self.mu_layer(out)
      log_sigma = self.log_sigma_layer(out)

      return mu, log_sigma

    else:
      return out

#  Self-attention module for deterministic path
class FeatureSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        return self.ln(x + attn_out)


#  Low-fidelity Neural Process
class LowFidelityNP(nn.Module):
    """
    Low-fidelity NP that additionally returns a sampled latent z_low.
    At train-time it samples z from the posterior; at test-time from the prior.
    Returns:
      - training: (mu, sigma, log_prob, kl, z)   where z shape = [B, num_targets, R_dim]
      - testing:  (mu, sigma, z)
    """
    def __init__(self, state_dim, control_dim, R_dim=128, num_heads=4, enable_ode_prior=False, device='cpu'):
        super().__init__()
        self.R_dim = R_dim
        self.device = device
        self.enable_ode_prior = enable_ode_prior

        self.combined_encoder = MLP(state_dim + control_dim + 1, R_dim, device=device)
        self.self_attn = FeatureSelfAttention(R_dim, num_heads)
        self.query_proj = nn.Linear(control_dim + 1, R_dim)
        self.decoder = MLP(2 * R_dim + control_dim + state_dim + 1, state_dim * 2, device=device)
        # latent encoder returns (mu, log_sigma) aggregated across context -> shapes [B, R_dim]
        self.latent_encoder = MLP(state_dim + control_dim + 1, R_dim,
                                  aggregate_step=True, num_latents=R_dim, device=device)

    def latent_path(self, context_x, context_y):
       # print(f'context_x: {torch.isnan(context_x).any()}, context_y: {torch.isnan(context_y).any()}')
        combined = torch.cat([context_y, context_x], dim=-1)  # [..., feat]
        mu, log_sigma = self.latent_encoder(combined)         # shapes: [B, R_dim]
        sigma = 0.9 * torch.sigmoid(log_sigma) + 0.1
     #   print(f'Latent mu: {torch.isnan(mu).any()}, sigma: {torch.isnan(sigma).any()}')
        return Normal(mu, sigma)                              # batch-distribution object

    def forward(self, query, pred_states, is_testing=False):
        (context_x, context_y), target_x = query
        B = target_x.shape[0]
        num_targets = target_x.shape[1]

        #print(f'BEFORE: context_x: {torch.isnan(context_x).any()}, context_y: {torch.isnan(context_y).any()}')

        prior = self.latent_path(context_x, context_y)  # Normal(mu_prior [B,R], sigma_prior [B,R])
        #print(f'pred_states: {torch.isnan(pred_states).any()}, target_x: {torch.isnan(target_x).any()}')
        #print(f'AFTER: context_x: {torch.isnan(context_x).any()}, context_y: {torch.isnan(context_y).any()}')
        if is_testing:
            # sample z_low from prior
            z_sample = prior.rsample()                              # [B, R_dim]
            z = z_sample.unsqueeze(1).repeat(1, num_targets, 1)     # [B, T, R_dim]
        else:
            posterior = self.latent_path(target_x, pred_states)    # conditioning on (target_x, pred_states)
            z_sample = posterior.rsample()                         # [B, R_dim]
            z = z_sample.unsqueeze(1).repeat(1, num_targets, 1)    # [B, T, R_dim]

        # deterministic representation with attention (same as before)
        rep = self.combined_encoder(torch.cat([context_y, context_x], dim=-1))
        rep = self.self_attn(rep)

        q = self.query_proj(target_x)  # [B, T, R_dim]
        attn_weights = F.softmax(torch.matmul(q, rep.transpose(-2, -1)) / (self.R_dim ** 0.5), dim=-1)
        deterministic_rep = torch.matmul(attn_weights, rep)  # [B, T, R_dim]

        combined_rep = torch.cat([deterministic_rep, z], dim=-1)  # [B, T, 2*R_dim]
        decoder_input = torch.cat([combined_rep, target_x, pred_states], dim=-1)  # [B, T, ...]
        decoder_out = self.decoder(decoder_input)
        mu, log_sigma = torch.chunk(decoder_out, 2, dim=-1)
        sigma = 0.9 * F.softplus(log_sigma) + 0.1

        dist = Independent(Normal(mu, sigma), 1)

        if not is_testing:
            kl = kl_divergence(posterior, prior).sum(-1, keepdim=True).repeat(1, num_targets)
            log_prob = dist.log_prob(pred_states)
            # return z (sampled from posterior) for hierarchical conditioning
            return mu, sigma, log_prob, kl, z
        # at test-time return sampled z from prior
        return mu, sigma, z


#  Residual High-fidelity Neural Process
class ResidualHighFidelityNP(nn.Module):
    """
    Residual NP that conditions its latent on the sampled low-fidelity latent z_low.
    Expectation: caller supplies `z_low` (sampled from LF model) with shape [B, T, R_dim]
    Forward signatures:
      - training (is_testing=False): returns mu, sigma, log_prob, kl   (kl computed between posterior and prior over z_high)
      - testing  (is_testing=True):  returns mu, sigma (and will use z_low as provided)
    NOTE: residual_y is still used for computing posterior during train.
    """
    def __init__(self, state_dim, control_dim, enable_ode_prior=False, R_dim=128, device='cpu'):
        super().__init__()
        self.R_dim = R_dim
        self.device = device
        self.enable_ode_prior = enable_ode_prior

        # latent encoder now conditions on context (x,y) and z_low
        # Input dim = (state_dim + control_dim + 1) + R_dim
        self.latent_encoder = MLP(state_dim + control_dim  + 1 + R_dim, R_dim,
                                  aggregate_step=True, num_latents=R_dim, device=device)

        # Decoder receives: z_high + z_low + target_x + context_y  (we keep format similar to earlier)
        # dims: R_dim (z_high) + R_dim (z_low) + control_dim+1 + state_dim
        self.decoder = MLP(2 * R_dim + control_dim + 1 + state_dim*2, state_dim * 2, device=device)

    def latent_path(self, context_x, context_y, z_low):
        """
        Build a latent distribution q(z_high | context, z_low)
        - context_x: [B, T_ctx, control_dim+perturb_dim+1]
        - context_y: [B, T_ctx, state_dim]
        - z_low:    [B, T_ctx, R_dim]  (should match the temporal dimension; will broadcast if needed)
        Returns a Normal(mu, sigma) with mu/sigma shapes [B, R_dim]
        """
        # concat along feature dim; ensure z_low has same sequence length as context_x/context_y
        if z_low.ndim == 2:
            # If z_low is [B, R_dim], broadcast to [B, context_len, R_dim]
            z_low_seq = z_low.unsqueeze(1).repeat(1, context_x.shape[1], 1)
        else:
            # if z_low [B, T, R], ensure T matches context length; otherwise try to trim/expand as needed
            if z_low.shape[1] != context_x.shape[1]:
                # If z_low is per-target but we're computing context prior (context_x has smaller T),
                # take the first timestep of z_low per-sample
                z_low_seq = z_low[:, :context_x.shape[1], :].contiguous()
            else:
                z_low_seq = z_low

        combined = torch.cat([context_y, context_x, z_low_seq], dim=-1)
        mu, log_sigma = self.latent_encoder(combined)  # returns [B, R_dim]
        sigma = 0.9 * torch.sigmoid(log_sigma) + 0.1
        return Normal(mu, sigma)

    def _apply_masking(self, residual_y, mask_p, unmask_ratio):
        """(unchanged) Mask only a fraction of batch entries based on unmask_ratio."""
        batch_size, _, _ = residual_y.shape
        masked_residual = residual_y.clone()

        if mask_p <= 0.0:
            # return consistent tuple
            return masked_residual, torch.zeros_like(residual_y, dtype=torch.bool, device=residual_y.device)

        num_unmasked = int(batch_size * unmask_ratio)
        unmasked_indices = torch.randperm(batch_size, device=residual_y.device)[:num_unmasked]

        mask = (torch.rand_like(residual_y) < mask_p)
        # unmask whole-sample rows
        for idx in unmasked_indices:
            mask[idx] = False

        masked_residual[mask] = 0.0
        return masked_residual, mask

    def forward(self, query, z_low, residual_y=torch.tensor([]), is_testing=False,
                mask_p=0.0, apply_mask=False, unmask_ratio=0.0, display_mask=False):
        """
        query: ((context_x, context_y), target_x)
        residual_y: [B, T, state_dim]  (used for posterior during training)
        z_low: either [B, R] (per-sample) or [B, T, R] (per-target) -- sampled from LowFidelityNP
        """
        (context_x, context_y), target_x = query
        B, num_targets, _ = target_x.shape

        '''
        # ----- Mask residual_y if requested (masking does NOT affect z_low) -----
        if apply_mask:
            masked_residual, mask = self._apply_masking(residual_y, mask_p, unmask_ratio)
            if display_mask:
                print("Residual mask (first 5):\n", mask[:5])
        else:
            masked_residual = residual_y
        '''

        # ----- Build prior and posterior conditioned on z_low -----
        # NOTE: we assume caller provides z_low that is compatible; if z_low is [B, R] or [B, T, R] we'll handle broadcasting.
        # For the prior we condition on context (context_x/context_y) and z_low.
        prior = self.latent_path(context_x, context_y[:, :context_x.shape[1], :], z_low)

        if is_testing:
            z_high_sample = prior.rsample()                             # [B, R_dim]
            z_high = z_high_sample.unsqueeze(1).repeat(1, num_targets, 1)  # [B, T, R_dim]
        else:
            # posterior conditions on target_x and masked_residual (which we treat as observed high-fidelity info)
            posterior = self.latent_path(target_x, residual_y, z_low)
            z_high_sample = posterior.rsample()                         # [B, R_dim]
            z_high = z_high_sample.unsqueeze(1).repeat(1, num_targets, 1)  # [B, T, R_dim]

        # ----- Decode: include both z_high and z_low as inputs -----
        # Ensure z_low per-target shape:
        if z_low.ndim == 2:
            z_low_targets = z_low.unsqueeze(1).repeat(1, num_targets, 1)  # [B, T, R_dim]
        else:
            # if z_low has time dim already, ensure same T as target_x
            if z_low.shape[1] != num_targets:
                z_low_targets = z_low[:, :num_targets, :].contiguous()
            else:
                z_low_targets = z_low

        decoder_input = torch.cat([z_high, z_low_targets, target_x, context_y, residual_y], dim=-1)
        decoder_out = self.decoder(decoder_input)
        mu, log_sigma = torch.chunk(decoder_out, 2, dim=-1)
        mu = F.tanh(mu)  # you used tanh earlier; keep that behavior
        sigma = 0.9 * F.softplus(log_sigma) + 0.1

        dist = Independent(Normal(mu, sigma), 1)

        if residual_y.any() and not is_testing:
            kl = kl_divergence(posterior, prior).sum(-1, keepdim=True).repeat(1, num_targets)
            log_prob = dist.log_prob(residual_y)
            return mu, sigma, log_prob, kl

        return mu, sigma
