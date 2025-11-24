#!/usr/bin/env python3
import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent, kl_divergence
from np_localization.np_model_utils import compute_state_priors

### General neural network structure for NP model ###
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
    
### 1D Convolutional for MHA (MultiHeaded Attention) within NP model ###
class Conv1D(nn.Module):
  def __init__(
      self,
      head_size=16,
      input_size=128,
      x_dim=1,
      stddev=1,
      device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  ):
    super(Conv1D, self).__init__()

    # Initializing kernel with normal distribution
    self.W = nn.init.normal_(torch.zeros([head_size, input_size, x_dim]), std=stddev).to(device)

  def forward(self, x):

    # Perform the 1D convolution
    out = F.conv1d(x.transpose(2, 1), self.W, bias=None)

    # Return the output by transposing back
    out.transpose_(2, 1)
    return out
  
### Attentive Neural Process (ANP) model structure ###
class AttLNP(nn.Module):

  def __init__(
      self,
      x_dim=3,
      y_dim=5,
      num_heads=8,
      att_type='mha',
      rep_transform = 'mlp',
      use_deterministic_path=True,
      R_dim=128,  # Size for encoded representation vector, R
      device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
      enable_ode_prior=False # Enables physics-informed mode
    ):

    super(AttLNP, self).__init__()
    self.device = device
    self.hidden_sizes = R_dim
    self.rep_transform = rep_transform
    self.att_type = att_type
    self.enable_ode_prior = enable_ode_prior
    self.use_deterministic_path = use_deterministic_path
    if self.att_type == 'mha'.lower():
      self.num_heads = num_heads

    # 1D Convolution Layers (for Multihead Attention)
    if self.att_type == 'mha':
      self.key_conv1d = Conv1D(device=self.device)
      self.value_conv1d = Conv1D(device=self.device)
      self.query_conv1d = Conv1D(device=self.device)
      self.rep_conv1d = Conv1D(
          head_size=R_dim, input_size=int(R_dim/num_heads), stddev=R_dim**(-0.5), device=self.device
      )

    # Encoder/Decoder MLPs
    encoder_size = x_dim + y_dim
    self.latent_encoder_network = MLP(
        input_size=encoder_size,
        output_size=R_dim,
        n_hidden_layers=2,
        aggregate_step=True,
        num_latents=R_dim,
        device=self.device
    )

    # Determine input size for decoder network
    if self.use_deterministic_path and self.enable_ode_prior: 
      decoder_input = 2*R_dim+x_dim+y_dim
    elif self.use_deterministic_path and not self.enable_ode_prior: 
      decoder_input = 2*R_dim+x_dim
    elif not self.use_deterministic_path and self.enable_ode_prior:
      decoder_input = R_dim+x_dim+y_dim
    else: 
      decoder_input = R_dim+x_dim
  
    self.decoder_network = MLP(
        input_size=decoder_input, # Input size for 128-dim representation vector + target_x (1-dim) concatenated
        output_size=y_dim*2,
        device=self.device
    )

    if self.use_deterministic_path:
      self.deterministic_encoder_network = MLP(
          input_size=encoder_size,
          output_size=R_dim,
          n_hidden_layers=2,
          device=self.device
      )

      # Key/Query MLPs
      if self.rep_transform == 'mlp'.lower():
        self.key_mlp = MLP(
            input_size=x_dim,
            output_size=R_dim,
            device=self.device
            )
        self.query_mlp = MLP(
            input_size=x_dim,
            output_size=R_dim,
            device=self.device
            )

  def latent_encoder(self, x, y):
    # Passing encoder input through MLP and deriving mean and log variance describing latent variable distribution
    encoder_input = torch.concat([x, y], dim=-1)
    mu, log_sigma = self.latent_encoder_network(encoder_input)

    # Bounding the log variance to obtain true variance values
    sigma = 0.9 * F.sigmoid(log_sigma) + 0.1

    return Normal(loc=mu, scale=sigma)

  def deterministic_encoder(self, context_x, context_y, target_x):

    num_context_points = context_x.shape[1]
    encoder_input = torch.concat([context_x, context_y], dim=-1)
    batch_size = encoder_input.shape[0]

    encoder_input = encoder_input.reshape([batch_size*num_context_points, -1])

    hidden = self.deterministic_encoder_network(encoder_input)
    v = hidden.reshape([batch_size, num_context_points, self.hidden_sizes])

    # Passing context and target inputs to form keys and query of attention model
    if self.rep_transform == 'mlp'.lower():
      k = self.key_mlp(context_x)
      q = self.query_mlp(target_x)
    elif self.rep_transform == 'identity'.lower():
      k, q = context_x, target_x

    # Calculating query-specific representation, R(*)
    if self.att_type == 'dpa'.lower(): # dot-product attention
      representation = F.scaled_dot_product_attention(q,k,v)
    elif self.att_type == 'mha'.lower(): # multi-headed attention
      representation = self.multihead_attention(q,k,v)
    elif self.att_type == 'ua'.lower(): # uniform attention
      representation = self.uniform_attention(q, v)

    return representation


  def decoder(self, representation, target_x, pred_states=torch.tensor([])):
    if pred_states.any():
      decoder_input = torch.concat([representation, target_x, pred_states], axis=-1)
    else:
      decoder_input = torch.concat([representation, target_x], axis=-1)
  
    num_total_points = target_x.shape[1]
    batch_size = decoder_input.shape[0]

    hidden = self.decoder_network(decoder_input)
    hidden = hidden.reshape([batch_size, num_total_points, -1])

    mu, log_sigma = torch.tensor_split(hidden, 2, dim=-1)

    sigma = 0.9 * F.softplus(log_sigma, threshold=100) + 0.1

    dist = Independent(Normal(loc=mu, scale=sigma), 1)

    return dist, mu, sigma


  def multihead_attention(self, q, k, v):
    """Computes multi-head attention.

    Args:
      q: queries. tensor of  shape [B,m,d_k].
      k: keys. tensor of shape [B,n,d_k].
      v: values. tensor of shape [B,n,d_v].
      num_heads: number of heads. Should divide d_v.

    Returns:
      tensor of shape [B,m,d_v].
    """
    batch_size = q.shape[0]
    query_size = q.shape[1]
    d_v = v.shape[-1]
    
    rep = torch.zeros([batch_size, query_size, d_v])
    for h in range(self.num_heads):
      # Shrinking dimension space of representations through 1D convolution
      k_conv = self.key_conv1d(k)
      q_conv = self.query_conv1d(q)
      v_conv = self.value_conv1d(v)

      # Gathering attention from reduced representations
      o = F.scaled_dot_product_attention(q_conv, k_conv, v_conv)
      rep += self.rep_conv1d(o)

    return rep

  def uniform_attention(self, q, v):
    # Equivalent to a LNP
    num_total_points = q.shape[1]
    rep = torch.mean(v, dim=1, keepdim=True)
    rep = torch.tile(rep, [1, num_total_points, 1])
    return rep

  def forward(
      self,
      query,
      input_unnorm=torch.tensor([]),
      target_y=torch.tensor([]),
      is_testing=False
  ):

    # Parsing data from dataset query
    (context_x, context_y), target_x = query
    context_x, context_y, target_x = context_x.to(self.device), context_y.to(self.device), target_x.to(self.device)
    num_total_points = target_x.shape[1]

    # Defining prior distribution over gaussian latent variable
    prior = self.latent_encoder(context_x, context_y)

    # For testing, when target_y unavailable, use contexts for latent encoder.
    if is_testing:
      latent_rep = prior.sample()

    # For training, when target_y is available, use targets for latent encoder.
    # Note that targets contain contexts by design.
    else:
      posterior = self.latent_encoder(target_x, target_y)
      latent_rep = posterior.sample()

    latent_rep = torch.tile(torch.unsqueeze(latent_rep, dim=1), [1, num_total_points, 1])

    if self.use_deterministic_path:
      deterministic_rep = self.deterministic_encoder(context_x, context_y, target_x)
      representation = torch.concat([deterministic_rep, latent_rep], dim=-1)
    else:
      representation = latent_rep
    
    if self.enable_ode_prior:
      pred_states = compute_state_priors(input_unnorm)
      dist, mu, sigma = self.decoder(representation, target_x, pred_states)

    else: 
      dist, mu, sigma = self.decoder(representation, target_x)

    # If we want to calculate the log_prob for training we will make use of the
    # target_y. At test time the target_y is not available so we return None.
    if target_y.any():
      posterior = self.latent_encoder(target_x, target_y)
      log_p = dist.log_prob(target_y)
      kl = torch.sum(
          kl_divergence(posterior, prior),
          dim=-1,
          keepdim=True
          )
      kl = torch.tile(kl, [1, num_total_points])
      
      return log_p, kl, mu, sigma

    else:

      return mu, sigma

