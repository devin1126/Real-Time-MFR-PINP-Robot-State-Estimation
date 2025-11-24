import torch
import torch.nn as nn

###   Function to load samples from CSV files   ###
def load_samples_from_csvs(data_dir: str, time_step: float, max_files: int | None = None):
    """
    Read CSV files in data_dir and build paired samples using a 2-row sliding window.
    Returns two tensors: samples_low, samples_high with shapes (N, 2, 8) where the
    8-dim row format is:
      [sampling_rate, left_wheel, right_wheel, twist_x, twist_y, ori_z, ori_w, thetadot]
    For the first row of each pair sampling_rate is 0.0, for the second row it is `time_step`.

    Behavior:
    - Files processed in sorted filename order (oldest first).
    - Each adjacent row pair (row[i], row[i+1]) yields one sample pair.
    - Missing/invalid numeric cells are skipped or filled from last valid value.
    - Non-finite samples are discarded.
    - Returns (samples_low, samples_high) as torch.float32 tensors.
    """
    import os
    import glob
    import csv
    import math
    import torch

    def _safe_float(val, fallback):
        try:
            v = float(val)
            return v if math.isfinite(v) else fallback
        except Exception:
            return fallback

    data_dir = os.path.expanduser(data_dir)
    pattern = os.path.join(data_dir, '*.csv')
    files = sorted(glob.glob(pattern))
    if max_files is not None:
        files = files[:max_files]

    samples_low = []
    samples_high = []

    # last seen fallbacks across files/rows
    last_left = None
    last_right = None
    last_low_twist_x = 0.0
    last_low_twist_y = 0.0
    last_low_ori_z = 0.0
    last_low_ori_w = 1.0
    last_low_tdot = 0.0
    last_high_twist_x = 0.0
    last_high_twist_y = 0.0
    last_high_ori_z = 0.0
    last_high_ori_w = 1.0
    last_high_tdot = 0.0

    for fname in files:
        try:
            with open(fname, 'r', newline='') as fh:
                reader = list(csv.DictReader(fh))
        except Exception:
            continue

        if len(reader) < 2:
            continue

        for i in range(len(reader) - 1):
            prev = reader[i]
            curr = reader[i + 1]

            # wheels (prefer explicit, fallback to last seen, then 0.0)
            fallback_left = last_left if last_left is not None else 0.0
            fallback_right = last_right if last_right is not None else 0.0
            prev_left = _safe_float(prev.get('wheel_left', prev.get('wheel_left', fallback_left)), fallback_left)
            prev_right = _safe_float(prev.get('wheel_right', prev.get('wheel_right', fallback_right)), fallback_right)
            curr_left = _safe_float(curr.get('wheel_left', prev_left), prev_left)
            curr_right = _safe_float(curr.get('wheel_right', prev_right), prev_right)

            # low-fidelity sources (imu_gz, imu_qz, imu_qw, twist_vx, twist_vy)
            prev_low_twist_x = _safe_float(prev.get('twist_vx', prev.get('twist_vx', last_low_twist_x)), last_low_twist_x)
            prev_low_twist_y = _safe_float(prev.get('twist_vy', prev.get('twist_vy', last_low_twist_y)), last_low_twist_y)
            curr_low_twist_x = _safe_float(curr.get('twist_vx', prev_low_twist_x), prev_low_twist_x)
            curr_low_twist_y = _safe_float(curr.get('twist_vy', prev_low_twist_y), prev_low_twist_y)

            prev_low_ori_z = _safe_float(prev.get('imu_qz', prev.get('im_qz', last_low_ori_z)), last_low_ori_z)
            prev_low_ori_w = _safe_float(prev.get('imu_qw', prev.get('im_qw', last_low_ori_w)), last_low_ori_w)
            curr_low_ori_z = _safe_float(curr.get('imu_qz', curr.get('im_qz', prev_low_ori_z)), prev_low_ori_z)
            curr_low_ori_w = _safe_float(curr.get('imu_qw', curr.get('im_qw', prev_low_ori_w)), prev_low_ori_w)

            prev_low_tdot = _safe_float(prev.get('imu_gz', prev.get('twist_wz', last_low_tdot)), last_low_tdot)
            curr_low_tdot = _safe_float(curr.get('imu_gz', curr.get('twist_wz', prev_low_tdot)), prev_low_tdot)

            # high-fidelity sources (odom_vx, odom_vy, odom_orien_z, odom_orien_w, odom_thetadot)
            prev_high_twist_x = _safe_float(prev.get('odom_vx', prev.get('odom_vx', last_high_twist_x)), last_high_twist_x)
            prev_high_twist_y = _safe_float(prev.get('odom_vy', prev.get('odom_vy', last_high_twist_y)), last_high_twist_y)
            curr_high_twist_x = _safe_float(curr.get('odom_vx', prev_high_twist_x), prev_high_twist_x)
            curr_high_twist_y = _safe_float(curr.get('odom_vy', prev_high_twist_y), prev_high_twist_y)

            prev_high_ori_z = _safe_float(prev.get('odom_orien_z', prev.get('odom_qz', last_high_ori_z)), last_high_ori_z)
            prev_high_ori_w = _safe_float(prev.get('odom_orien_w', prev.get('odom_qw', last_high_ori_w)), last_high_ori_w)
            curr_high_ori_z = _safe_float(curr.get('odom_orien_z', curr.get('odom_qz', prev_high_ori_z)), prev_high_ori_z)
            curr_high_ori_w = _safe_float(curr.get('odom_orien_w', curr.get('odom_qw', prev_high_ori_w)), prev_high_ori_w)

            prev_high_tdot = _safe_float(prev.get('odom_thetadot', prev.get('twist_wz', last_high_tdot)), last_high_tdot)
            curr_high_tdot = _safe_float(curr.get('odom_thetadot', curr.get('twist_wz', prev_high_tdot)), prev_high_tdot)

            # Build numeric rows in required order
            row_prev_low = [
                0.0,                  # sampling rate placeholder for prev row
                prev_left,
                prev_right,
                prev_low_twist_x,
                prev_low_twist_y,
                prev_low_ori_z,
                prev_low_ori_w,
                prev_low_tdot
            ]
            row_curr_low = [
                float(time_step),
                0.0, 0.0,
                curr_low_twist_x,
                curr_low_twist_y,
                curr_low_ori_z,
                curr_low_ori_w,
                curr_low_tdot
            ]

            row_prev_high = [
                0.0,
                prev_left,
                prev_right,
                prev_high_twist_x,
                prev_high_twist_y,
                prev_high_ori_z,
                prev_high_ori_w,
                prev_high_tdot
            ]
            row_curr_high = [
                float(time_step),
                0.0, 0.0,
                curr_high_twist_x,
                curr_high_twist_y,
                curr_high_ori_z,
                curr_high_ori_w,
                curr_high_tdot
            ]

            # Convert to torch and ensure finite
            t1 = torch.tensor(row_prev_low, dtype=torch.float32)
            t2 = torch.tensor(row_curr_low, dtype=torch.float32)
            h1 = torch.tensor(row_prev_high, dtype=torch.float32)
            h2 = torch.tensor(row_curr_high, dtype=torch.float32)

            if not torch.isfinite(t1).all() or not torch.isfinite(t2).all() or not torch.isfinite(h1).all() or not torch.isfinite(h2).all():
                # skip any window with invalid numbers
                continue

            samples_low.append(torch.stack([t1, t2], dim=0))
            samples_high.append(torch.stack([h1, h2], dim=0))

            # update last-seen values
            last_left = curr_left
            last_right = curr_right
            last_low_twist_x = curr_low_twist_x
            last_low_twist_y = curr_low_twist_y
            last_low_ori_z = curr_low_ori_z
            last_low_ori_w = curr_low_ori_w
            last_low_tdot = curr_low_tdot
            last_high_twist_x = curr_high_twist_x
            last_high_twist_y = curr_high_twist_y
            last_high_ori_z = curr_high_ori_z
            last_high_ori_w = curr_high_ori_w
            last_high_tdot = curr_high_tdot

    if len(samples_low) == 0:
        return torch.empty((0, 2, 8), dtype=torch.float32), torch.empty((0, 2, 8), dtype=torch.float32)

    samples_low_t = torch.stack(samples_low, dim=0)   # (N,2,8)
    samples_high_t = torch.stack(samples_high, dim=0) # (N,2,8)

    return samples_low_t, samples_high_t

###   Minmax normalization function   ###
def minmax_norm(dataset):

    minmax_data = torch.tensor([[0,0],
                                [-25,25],
                                [-25,25],
                                [-1.0,1.0],
                                [-1.0,1.0],
                                [0,0],
                                [0,0],
                                [-2,2]])

    # Parsing relevant data shapes
    dynamical_data_dim = dataset.shape[2]
    prev_dim = 0
    next_dim = 1
    prev_dim_skips = [0] + list(range(5,7))
    next_dim_skips = list(range(3)) + list(range(5,7))

    for dim in range(dynamical_data_dim):
        # Skipping sampling rate dimension to avoid normalization
        if dim in prev_dim_skips:
            continue

        curr_sequence = dataset[:,prev_dim,dim:dim+1]
        curr_min = minmax_data[dim,0]
        curr_max = minmax_data[dim,1]

        dataset[:,prev_dim,dim:dim+1] = (curr_sequence - curr_min) / (curr_max - curr_min)
        
        
    for dim in range(dynamical_data_dim):
        # Skipping sampling rate dimension to avoid normalization
        if dim in next_dim_skips:
            continue

        curr_sequence = dataset[:,next_dim,dim:dim+1]
        curr_min = minmax_data[dim,0]
        curr_max = minmax_data[dim,1]

        dataset[:,next_dim,dim:dim+1] = (curr_sequence - curr_min) / (curr_max - curr_min)
      
    return dataset

###   Minmax unnormalization function   ###
def minmax_unnorm(pred_y, var, target_y=torch.tensor([])):

    minmax_data = torch.tensor([[0,0],
                                [-25,25],
                                [-25,25],
                                [-1.0,1.0],
                                [-1.0,1.0],
                                [0,0],
                                [0,0],
                                [-2,2]])

    state_dim = pred_y.shape[2]
    dim_skips = [0] + list(range(5,7))

    # Unnormalizing input array trajectory-by-trajectory
    for i,dim in enumerate(range(3, 3+state_dim)):
        if dim in dim_skips:
            continue

        min_value = minmax_data[dim, 0]
        max_value = minmax_data[dim, 1]

        if target_y.any(): target_y[:,:,i] = target_y[:,:,i] * (max_value - min_value) + min_value
        pred_y[:,:,i] = pred_y[:,:,i] * (max_value - min_value) + min_value
        var[:,:,i] = torch.sqrt(var[:,:,i] * (max_value - min_value))

    if target_y.any(): 
       return pred_y, var, target_y

    return pred_y, var

###   ELBO Loss function for NP models  ###
class ELBO(nn.Module):
  def __init__(self):
    super(ELBO, self).__init__()

  def forward(self, log_p, kl):
    # Get the device of log_p (assuming both log_p and kl should be on the same device)
    device = log_p.device
    
    # Ensure that both tensors are on the same device
    log_p = log_p.to(device)
    kl = kl.to(device)
    
    # Computing total loss
    loss = -torch.mean((log_p - kl))
    
    return loss
  

###   Function that randomizes order of first data dimension within given dataset   ###
def index_data_stacker(dataset):
    # Parsing relevant data shapes
    num_targets = dataset.shape[0]
    traj_length = dataset.shape[1]
    stack_iters = dataset.shape[2]

    # Initializing index data objects needed for computation
    idx = torch.randperm(num_targets)
    idx = torch.unsqueeze(idx, dim=-1)
    idx_array = torch.tensor([])
    idx_tnsr = torch.zeros([num_targets, traj_length, stack_iters])
    
    # Ensuring idx_array.shape == idx_tnsr[:,:,i].shape 
    for _ in range(traj_length):
      idx_array = torch.concat([idx_array, idx], dim=1)
    idx_array = torch.unsqueeze(idx_array, dim=-1)

    # Updating each dimension of idx_tnsr with idx_array
    for stack in range(stack_iters):
      idx_tnsr[:,:,stack:stack+1] = idx_array

    return idx_tnsr.to(dtype=torch.int64) # ensure all entries are integers


###  Function that computes the Average Absolute Error (AAE) between target and predicted odometry states  ###
def compute_AAE(target_y, pred_y):

  # Computing absolute prediction error (AE)
  state_diff = torch.abs(target_y[:,-1,:] - pred_y[:,-1,:])
  AAE_array = torch.mean(torch.sum(state_diff, axis=1))
  
  return AAE_array

###  Function that computes the Root Mean Squared Error (RMSE) between target and predicted odometry states  ###
def compute_RMSE(target_y, pred_y):
    """
    Compute the batch-averaged sum of RMSEs for each state at the final time step.
    
    Args:
        target_y (torch.Tensor): Shape (batch_size, time_steps, state_dim)
        pred_y (torch.Tensor): Shape (batch_size, time_steps, state_dim)
    
    Returns:
        float: Scalar RMSE value averaged over the batch.
    """
    # Compute squared error at the final time step
    state_diff_squared = (target_y[:, -1, :] - pred_y[:, -1, :]) ** 2  # Shape: (batch_size, state_dim)
    
    # RMSE for each state in each sample
    rmse_per_state = torch.sqrt(state_diff_squared)  # Shape: (batch_size, state_dim)

    # Sum RMSE across states for each sample
    summed_rmse_per_sample = torch.sum(rmse_per_state, axis=1)  # Shape: (batch_size,)

    # Compute average and standard deviation of RMSE across the batch
    batch_rmse = torch.mean(summed_rmse_per_sample)
    batch_rmse_std = torch.std(summed_rmse_per_sample)

    return batch_rmse, batch_rmse_std

###  Function that computes the Average Negative Log-Likelihood (NLL) between target and predicted odometry states  ###
def compute_avg_nll(target_y, pred_y, var):
    """
    Compute the Average Negative Log-Likelihood (NLL) for a batch of data.

    Args:
    - target_y (torch.Tensor): Ground truth states of shape [batch_size, num_states].
    - pred_y (torch.Tensor): Predicted means of shape [batch_size, num_states].
    - var (torch.Tensor): Predicted variances of shape [batch_size, num_states].

    Returns:
    - avg_nll (torch.Tensor): Average Negative Log-Likelihood value.
    """
    # Ensure that variances are positive (avoid numerical issues with zero/negative variances)
    epsilon = 1e-8  # Small value to prevent division by zero
    var = torch.maximum(var, torch.tensor(epsilon, device=var.device))  # Make sure variance is positive

    # Calculate the NLL for each state in the batch
    nll_per_state = 0.5 * torch.log(2 * torch.pi * var) + 0.5 * ((target_y - pred_y) ** 2) / var

    # Compute the average NLL over all samples and states
    avg_nll = torch.mean(nll_per_state)

    return avg_nll

###   Runge-Kutta 4th order integration function   ###
def runge_kutta_4(pose, velocity, quat, dt):
    """
    Perform 4th-order Runge-Kutta integration to update the pose.
    :param pose: Current position as a tensor [x, y].
    :param velocity: Linear velocity as a tensor [vx, vy].
    :param angular_velocity: Angular velocity as a scalar tensor.
    :param dt: Time step for integration.
    :return: Updated pose as a tensor [x, y].
    """
    def dynamics(V, theta):
        return torch.tensor([V*torch.cos(theta), V*torch.sin(theta)])

    V = torch.norm(velocity)
    theta = torch.atan2(quat[0], quat[1])
    k1 = dynamics(V,theta)
    k2 = dynamics(V,theta)
    k3 = dynamics(V,theta)
    k4 = dynamics(V,theta)
    
   # print(f"pose: {pose}, velocity: {velocity}, quat: {quat}, dt: {dt}")
   # print(f"k1: {k1}, k2: {k2}, k3: {k3}, k4: {k4}")
    return pose + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


###   Function that converts yaw angle to quaternion   ###
def theta2quat(yaw_batch):
    # Assume roll and pitch are 0 for all samples in the batch
    roll_batch = torch.zeros_like(yaw_batch, dtype=yaw_batch.dtype)  # batch_size
    pitch_batch = torch.zeros_like(yaw_batch, dtype=yaw_batch.dtype)  # batch_size
    
    # Initialize the quaternion tensor to hold batch quaternions
    q_batch = torch.zeros((yaw_batch.size(0), 2), device=yaw_batch.device, dtype=yaw_batch.dtype)

    # Compute the quaternion components for each sample in the batch
    q_batch[:, 0] = torch.cos(roll_batch / 2) * torch.cos(pitch_batch / 2) * torch.sin(yaw_batch / 2) - \
                    torch.sin(roll_batch / 2) * torch.sin(pitch_batch / 2) * torch.cos(yaw_batch / 2)
    
    q_batch[:, 1] = torch.cos(roll_batch / 2) * torch.cos(pitch_batch / 2) * torch.cos(yaw_batch / 2) + \
                    torch.sin(roll_batch / 2) * torch.sin(pitch_batch / 2) * torch.sin(yaw_batch / 2)

    # Normalize the quaternion to ensure it has unit norm for each batch element
    qmag = torch.sqrt(torch.sum(q_batch ** 2, dim=1, keepdim=True))  # shape: (batch_size, 1)
    q_batch /= qmag

    return q_batch


###   Function that converts quaternion to yaw angle   ###
def quat2theta(q_batch):
    # Work on a copy so we don't mutate the caller
    q = q_batch.clone().to(dtype=q_batch.dtype)

    eps = 1e-8
    # mask True for rows that are (approximately) all zeros
    zero_mask = (q.abs() <= eps).all(dim=1)  # shape: (batch,)

    if zero_mask.any().item():
        # set zero rows to safe default quaternion [z=0, w=1]
        q[zero_mask] = torch.tensor([0.0, 1.0], device=q.device, dtype=q.dtype)

    # safe normalization (avoid div-by-zero)
    qmag = torch.sqrt(torch.sum(q * q, dim=1, keepdim=True))  # (batch,1)
    qmag_safe = torch.where(qmag > eps, qmag, torch.ones_like(qmag))
    q = q / qmag_safe

    # map (z,w) -> yaw
    q0 = q[:, 1]
    q3 = q[:, 0]
    # roll/pitch assumed zero so yaw = atan2(2*(q0*q3), 1 - 2*(q3**2))
    yaw_batch = torch.atan2(2 * (q0 * q3), 1 - 2 * (q3 ** 2))

    return yaw_batch


###   Function that computes the marginal conformal quantiles for each state   ###
def compute_marginal_conformal_quantiles(target_y, pred_y, var, alpha=0.95, batch_divide=100):
    """
    Computes the conformal quantiles for the nonconformity scores of each state in the dynamical system.

    Args:
    - target_y (torch.Tensor): True target states of the system (batch_size, time_steps, num_states).
    - pred_y (torch.Tensor): Predicted states of the system (batch_size, time_steps, num_states).
    - var (torch.Tensor): Model variance for each state (batch_size, time_steps, num_states, num_states).
    - alpha (float): The quantile value to compute, default is 0.95.
    - batch_divide (int): Number of batches to divide the computation (default is 100).

    Returns:
    - quantiles (torch.Tensor): The computed quantiles for each state (num_states,).
    """
    num_states = target_y.shape[2]  # The number of states corresponds to the size of the last dimension
    nonconformity_scores_per_state = torch.zeros((num_states, target_y.shape[0]))  # (num_states, batch_size)

    # Compute nonconformity scores for each state individually
    for i in range(num_states):
        # Extract the data for the current state (index 2 corresponds to states)
        target_state = target_y[:, -1, i]  # Shape: (batch_size,)
        pred_state = pred_y[:, -1, i]      # Shape: (batch_size,)
        var_state = var[:, -1, i]          # Shape: (batch_size,) (the variance of state i)

        # Compute Mahalanobis distance for each state
        batch_len = int(var_state.size(0) / batch_divide) if var_state.size(0) >= batch_divide else var_state.size(0)
        nonconformity_scores = torch.zeros(var_state.shape[0])

        for j in range(0, var_state.shape[0], batch_len):

            # Slice batch data
            cov_batch = torch.diag_embed(var_state[j:j+batch_len])  # Covariance matrix for each batch
            x_true_batch = target_state[j:j+batch_len]  # True target values
            x_pred_batch = pred_state[j:j+batch_len]    # Predicted values

            # Compute Mahalanobis distance for the batch
            diff = x_true_batch - x_pred_batch  # Shape: (batch_len,)
            diff = diff.unsqueeze(-1)  # Shape: (batch_len, 1)
            inv_covs = torch.linalg.inv(cov_batch)  # Shape: (batch_len, batch_len)

            # Mahalanobis distance calculation using matrix multiplication
            mahalanobis_dist = torch.sqrt(torch.sum(diff * torch.matmul(inv_covs, diff), dim=1))  # Shape: (batch_len,)

            # Store nonconformity scores in the array for this state
            nonconformity_scores[j:j+batch_len] = mahalanobis_dist

        # Store nonconformity scores for the current state
        nonconformity_scores_per_state[i, :] = nonconformity_scores

    # Compute quantiles for each state separately
    quantiles = torch.quantile(nonconformity_scores_per_state, alpha, dim=1)  # Shape: (num_states,)

    return quantiles


###   Function that computes the forward ODE state priors for the low fidelity NP model   ###
def compute_state_priors(input_unnorm, dt=0.034):

    ###### NESTED FUNCTIONS START ######

    # Defining system dynamics for batch processing
    def jacobian_batch(theta_batch, wheel_rad=0.033, chassis_width=0.288):
        # We expect theta_batch to be of shape [batch_size]
        cx = torch.cos(theta_batch)
        sx = torch.sin(theta_batch)
        
        # Initialize Jacobian matrix for the batch
        J = torch.zeros([theta_batch.size(0), 3, 2], dtype=theta_batch.dtype)
        
        # Populate the Jacobian for each robot in the batch
        J[:, 0, 0] = (wheel_rad / 2) * cx  # J[0, 0] for each batch element
        J[:, 0, 1] = (wheel_rad / 2) * cx  # J[0, 1] for each batch element
        J[:, 1, 0] = (wheel_rad / 2) * sx  # J[1, 0] for each batch element
        J[:, 1, 1] = (wheel_rad / 2) * sx  # J[1, 1] for each batch element
        J[:, 2, 0] = -wheel_rad / chassis_width  # J[2, 0] for each batch element
        J[:, 2, 1] = wheel_rad / chassis_width  # J[2, 1] for each batch element

        return J
    
    def normalize_states(pred_states):
        minmax_data = torch.tensor([[-1.0,1.0],
                                    [-1.0,1.0],
                                    [0,0],
                                    [0,0],
                                    [-2,2]], dtype=pred_states.dtype)
        
        dim_skips = list(range(2,4))
        for dim in range(pred_states.shape[2]):
            if dim in dim_skips:
                continue
        
            curr_sequence = pred_states[:,:,dim:dim+1]
            curr_min = minmax_data[dim,0]
            curr_max = minmax_data[dim,1]

            pred_states[:,:,dim:dim+1] = (curr_sequence - curr_min) / (curr_max - curr_min)
        
        return pred_states
    
    ###### NESTED FUNCTIONS END ######

    # Parsing the navigation states
    Xdot_prev = input_unnorm[:, 0, 3]  # Position in X (batch_size,)
    Ydot_prev = input_unnorm[:, 0, 4]  # Position in Y (batch_size,)
    q_prev = input_unnorm[:, 0, 5:7]  # Orientation (batch_size,2)
    omega_prev = input_unnorm[:, 0, 7]  # Angular velocity (batch_size,)
    yaw_batch = quat2theta(q_prev.clone())  # Convert quaternion to yaw (batch_size,)

    #print(f'Xdot_prev: {torch.isnan(Xdot_prev).any()}, Ydot_prev: {torch.isnan(Ydot_prev).any()}, q_prev: {torch.isnan(q_prev).any()}, omega_prev: {torch.isnan(omega_prev).any()}, yaw_batch: {torch.isnan(yaw_batch).any()}')
    # Compute the Jacobian for the batch of theta values
    J_batch = jacobian_batch(yaw_batch)

    #print(f'input_unnorm (first ts): {input_unnorm[:,0,:]}')
    #print(f'input_unnorm (second ts): {input_unnorm[:,1,:]}')

    # Compute the linear and angular velocities for each robot in the batch
    uu_batch = input_unnorm[:, 0, 1:3]
    #print(f'uu_batch: {uu_batch}')
    vel_batch = torch.bmm(J_batch, uu_batch.unsqueeze(2)).squeeze(-1)  # Batched matrix multiplication
    Xdot_next = vel_batch[:, 0]  # Linear velocity in X (batch_size,)
    Ydot_next = vel_batch[:, 1]  # Linear velocity in Y (batch_size,)
    omega_next = vel_batch[:, 2]  # Angular velocity (batch_size,)

   # print(f'Xdot_next: {Xdot_next}, Ydot_next: {Ydot_next}, omega_next: {omega_next}')

    yaw_batch += omega_next * dt  # Update the orientation
    q_next = theta2quat(yaw_batch)  # Convert the updated yaw to quaternion

    # Forming predictive states for NP decoder
    tensor1 = torch.cat([Xdot_prev.unsqueeze(1), Ydot_prev.unsqueeze(1), q_prev, omega_prev.unsqueeze(1)], dim=1).unsqueeze(1)
    tensor2 = torch.cat([Ydot_next.unsqueeze(1), Xdot_next.unsqueeze(1), q_next[:,0].unsqueeze(1), q_next[:,1].unsqueeze(1), omega_next.unsqueeze(1)], dim=1).unsqueeze(1)
    pred_states = torch.cat([tensor1, tensor2], dim=1)

    # Normalize the predictive states
    pred_states = normalize_states(pred_states)
   # print(f'pred_states: {pred_states}')

    return pred_states


###   Function that computes the forward ODE state priors for the residual high fidelity NP model   ###
def compute_residual_priors(input_unnorm, dt=0.034):

    """
    Tier 2 dynamic model for a skid-steering differential drive robot.
    Computes next-step state priors with simple slip dynamics.
    """

    def normalize_states(pred_states):
        minmax_data = torch.tensor([[-1.0,1.0],
                                    [-1.0,1.0],
                                    [0,0],
                                    [0,0],
                                    [-2,2]])
        
        dim_skips = list(range(2,4))
        for dim in range(pred_states.shape[2]):
            if dim in dim_skips:
                continue
        
            curr_sequence = pred_states[:,:,dim:dim+1]
            curr_min = minmax_data[dim,0]
            curr_max = minmax_data[dim,1]

            pred_states[:,:,dim:dim+1] = (curr_sequence - curr_min) / (curr_max - curr_min)
        
        return pred_states

    # --- Parameters ---
    wheel_rad = 0.033       # [m]
    half_wheelbase = 0.144  # [m] (half of 0.288)
    m = 8.0                 # robot mass [kg]
    Iz = 0.25               # yaw inertia [kg·m^2]
    Ct = 60.0               # traction coefficient (N·s/m)
    Calpha = 40.0           # lateral slip coefficient (N·s/m)

    # --- Extract current states ---
    Xdot_prev = input_unnorm[:, 0, 3]
    Ydot_prev = input_unnorm[:, 0, 4]
    q_prev = input_unnorm[:, 0, 5:7]
    omega_prev = input_unnorm[:, 0, 7]
    yaw_batch = quat2theta(q_prev)

    # --- Wheel commands [v_l, v_r] ---
    uu_batch = input_unnorm[:, 0, 1:3]
    v_l = uu_batch[:, 0]
    v_r = uu_batch[:, 1]

    # --- Compute wheel-ground longitudinal forces ---
    Fx_left = Ct * v_l
    Fx_right = Ct * v_r

    # --- Aggregate body-frame longitudinal/lateral/yaw forces ---
    Fx = 0.5 * (Fx_left + Fx_right)
    Fy = -Calpha * Ydot_prev       # resistive lateral force
    Mz = half_wheelbase * (Fx_right - Fx_left)

    # --- Integrate body-frame velocities ---
    u_prev = Xdot_prev
    v_prev = Ydot_prev
    r_prev = omega_prev

    u_dot = (Fx - m * v_prev * r_prev) / m
    v_dot = (Fy + m * u_prev * r_prev) / m
    r_dot = Mz / Iz

    u_next = u_prev + u_dot * dt
    v_next = v_prev + v_dot * dt
    r_next = r_prev + r_dot * dt

    # --- Update orientation and world-frame motion ---
    yaw_next = yaw_batch + r_next * dt
    X_next = u_next * torch.cos(yaw_batch) - v_next * torch.sin(yaw_batch)
    Y_next = u_next * torch.sin(yaw_batch) + v_next * torch.cos(yaw_batch)
    q_next = theta2quat(yaw_next)

    # --- Form predictive states for NP decoder ---
    tensor1 = torch.cat([
        Xdot_prev.unsqueeze(1), 
        Ydot_prev.unsqueeze(1),
        q_prev,
        omega_prev.unsqueeze(1)
    ], dim=1).unsqueeze(1)

    tensor2 = torch.cat([
        Y_next.unsqueeze(1),
        X_next.unsqueeze(1),
        q_next[:, 0].unsqueeze(1),
        q_next[:, 1].unsqueeze(1),
        r_next.unsqueeze(1)
    ], dim=1).unsqueeze(1)

    pred_states = torch.cat([tensor1, tensor2], dim=1)

    # --- Normalize states (reuse your normalization function) ---
    pred_states = normalize_states(pred_states)

    return pred_states



###   NP model training function   ###
def train_np_model(model_low, model_res, replay_buffer, optimizer, loss_function, batch_size, model_low_target, enable_ode_prior=False):
    # Setting the model to training mode
    model_low.train()
    model_res.train()

    # Randomly sample a batch of data from the replay buffer
    batch = [replay_buffer[i] for i in torch.randint(0, len(replay_buffer), (batch_size,))]

    # Separate the batch into low and high datasets
    batch_low = torch.stack([item[0] for item in batch]).squeeze(1)  # Extract sample_low
    batch_high = torch.stack([item[1] for item in batch]).squeeze(1)  # Extract sample_high

    # Normalizing batch input tensor for the NP model    
    norm_batch_low = minmax_norm(batch_low.clone())
    norm_batch_high = minmax_norm(batch_high.clone())

    # Defining LF context/target sets for the NP model
    context_x = norm_batch_low[:,:1,:3].clone()
    context_y_low = norm_batch_low[:,:1,3:].clone()
    target_x = norm_batch_low[:,:,:3].clone()
    target_y = norm_batch_high[:,:,3:].clone()

    # Initializing query object for model
    query_low = ((context_x, context_y_low), target_x)

    # Predicting the next odometry state and capturing loss info
   # print(f'batch_low: {torch.isnan(batch_low).any()}, batch_high: {torch.isnan(batch_high).any()}')
    pred_states = compute_state_priors(batch_low, dt=target_x[0,-1,0].item())
   # print(f'pred_states: {torch.isnan(pred_states).any()}')
    mu_low, sigma_low, log_p_low, kl_low, z_low = model_low(query_low, pred_states=pred_states, is_testing=False)

    # Constructing residual labels for the high fidelity NP model
    with torch.no_grad():
        mu_low_target, _, _ = model_low_target(query_low, pred_states=pred_states, is_testing=True)

    residual_label = target_y - mu_low_target.detach()

    # Predicting the residual dynamcis and capturing loss info
    query_res = ((context_x, mu_low.detach()), target_x)
    mu_res, sigma_res, log_p_res, kl_res = model_res(
        query_res, 
        residual_y=residual_label, 
        z_low=z_low.detach(), 
        is_testing=False
    )

    # Fusing the low fidelity and residual model predictions
    fused_mu = mu_low + mu_res
    fused_var = sigma_low**2 + sigma_res**2

    # Calculate the loss and providing recurrent performance feedback
    loss_low = loss_function(log_p_low, kl_low)
    loss_res = loss_function(log_p_res, kl_res)
    loss = loss_low + loss_res

    # Backpropagate and optimize
    optimizer.zero_grad()  # Clear gradients from previous step
    loss.backward()  # Compute gradients
    optimizer.step()  # Update the model weights

    # Compute batch average absolute error (AAE)
    fused_mu, fused_var, target_y = minmax_unnorm(fused_mu, fused_var, target_y)
    q_marginal = compute_marginal_conformal_quantiles(target_y, fused_mu, fused_var, alpha=0.95)
    RMSE, RMSE_std = compute_RMSE(target_y, fused_mu)
    NLL = compute_avg_nll(target_y[:,-1,:], fused_mu[:,-1,:], fused_var[:,-1,:])
    fused_var[:,-1,:] *= q_marginal.unsqueeze(0)
    NLL_marginal = compute_avg_nll(target_y[:,-1,:], fused_mu[:,-1,:], fused_var[:,-1,:])

    return model_low, model_res, loss, RMSE, RMSE_std, NLL, NLL_marginal, q_marginal