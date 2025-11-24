import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import glob
import csv
import math
import torch

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

    minmax_data = torch.tensor([[-1.0,1.0],
                                [-1.0,1.0],
                                [0,0],
                                [0,0],
                                [-2,2],
                                [0,0],
                                [-25,25],
                                [-25,25],
                                [-1.0,1.0],
                                [-1.0,1.0],
                                [0,0],
                                [0,0],
                                [-2,2]])

    # Parsing relevant data shapes
    dynamical_data_dim = dataset.shape[1]
    dim_skips = list(range(2,4)) + [5] + list(range(5,7)) + list(range(10,12))

    for dim in range(dynamical_data_dim):
        # Skipping sampling rate dimension to avoid normalization
        if dim in dim_skips:
            continue

        curr_sequence = dataset[:,dim:dim+1]
        curr_min = minmax_data[dim,0]
        curr_max = minmax_data[dim,1]

        dataset[:,dim:dim+1] = (curr_sequence - curr_min) / (curr_max - curr_min)
        
        
    return dataset

###   Minmax unnormalization function   ###
def minmax_unnorm(pred_state_vecs, Pz, true_state_vecs=torch.tensor([])):

    minmax_data = torch.tensor([[-1.0,1.0],
                                [-1.0,1.0],
                                [0,0],
                                [0,0],
                                [-2,2]])

    state_dim = pred_state_vecs.shape[1]
    dim_skips = list(range(2,4))

    # Unnormalizing input array trajectory-by-trajectory
    for dim in range(state_dim):
        if dim in dim_skips:
            continue

        min_value = minmax_data[dim, 0]
        max_value = minmax_data[dim, 1]

        if true_state_vecs.any(): true_state_vecs[:,dim] = true_state_vecs[:,dim] * (max_value - min_value) + min_value
        pred_state_vecs[:,dim] = pred_state_vecs[:,dim] * (max_value - min_value) + min_value
        Pz[:,dim,dim] = torch.sqrt(Pz[:,dim,dim] * (max_value - min_value))

    if true_state_vecs.any(): 
       return pred_state_vecs, Pz, true_state_vecs
    
    return pred_state_vecs, Pz

###   DKF Loss function   ###
class DKFLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, gamma=5e-3):
        super(DKFLoss, self).__init__()
        self.alpha = alpha  # Weight for reconstruction loss
        self.beta = beta    # Weight for measurement prediction loss
        self.gamma = gamma  # Weight for residual loss

    def forward(self, z_pred, z_true, x_pred, x_true):

        # Observed state prediction loss (MSE)
        measure_loss = F.mse_loss(x_pred, x_true)

        # Latent state prediction loss (MSE)
        latent_loss = F.mse_loss(z_pred, z_true)

        # Mahalanobis distance computations for state covariances
        #mal_dists = batch_mahalanobis_distance(R_x, P_z, x_true, z_true)
        
        # Total loss
        total_loss = self.alpha * measure_loss + self.beta * latent_loss #+ self.gamma * residual_loss

        return total_loss
        
  

###   Function that randomizes order of first data dimension within given dataset   ###
def index_data_stacker(dataset):
    # Parsing relevant data shapes
    num_targets = dataset.shape[0]
    stack_iters = dataset.shape[1]

    # Initializing index data objects needed for computation
    idx = torch.randperm(num_targets)
    idx = torch.unsqueeze(idx, dim=-1)
    idx_tnsr = torch.zeros([num_targets, stack_iters])

    # Updating each dimension of idx_tnsr with idx_array
    for stack in range(stack_iters):
      idx_tnsr[:,stack:stack+1] = idx

    return idx_tnsr.to(dtype=torch.int64) # ensure all entries are integers


###  Function that computes the Average Absolute Error (AAE) between target and predicted odometry states  ###
def compute_AAE(target_y, pred_y):

  # Computing absolute prediction error (AE)
  state_diff = torch.abs(target_y[:,-1,:] - pred_y[:,-1,:])
  AAE_array = torch.mean(torch.sum(state_diff, axis=1))
  
  return AAE_array

###  Function that computes the Root Mean Squared Error (RMSE) between target and predicted odometry states  ###
def compute_RMSE(true_state_vecs, pred_state_vecs):
    """
    Compute the batch-averaged sum of RMSEs for each state at the final time step.
    
    Args:
        true_state_vecs (torch.Tensor): Shape (batch_size, time_steps, state_dim)
        pred_state_vecs (torch.Tensor): Shape (batch_size, time_steps, state_dim)
    
    Returns:
        float: Scalar RMSE value averaged over the batch.
    """
    # Compute squared error at the final time step
    state_diff_squared = (true_state_vecs - pred_state_vecs) ** 2  # Shape: (batch_size, state_dim)
    
    # RMSE for each state in each sample
    rmse_per_state = torch.sqrt(state_diff_squared)  # Shape: (batch_size, state_dim)

    # Sum RMSE across states for each sample
    summed_rmse_per_sample = torch.sum(rmse_per_state, axis=1)  # Shape: (batch_size,)

    # Compute average and standard deviation of RMSE across the batch
    batch_rmse = torch.mean(summed_rmse_per_sample)
    batch_rmse_std = torch.std(summed_rmse_per_sample)

    return batch_rmse, batch_rmse_std

###  Function that computes the Average Negative Log-Likelihood (NLL) between target and predicted odometry states  ###
def compute_avg_nll(target_y, pred_y, Pz):
    """
    Compute the Average Negative Log-Likelihood (NLL) for a batch of data.

    Args:
    - target_y (torch.Tensor): Ground truth states of shape [batch_size, num_states].
    - pred_y (torch.Tensor): Predicted means of shape [batch_size, num_states].
    - Pz (torch.Tensor): Predicted covariance matrices of shape [batch_size, num_states, num_states].

    Returns:
    - avg_nll (torch.Tensor): Average Negative Log-Likelihood value.
    """

    # Extract variances from the diagonal of the covariance matrices
    var = torch.diagonal(Pz, offset=0, dim1=1, dim2=2)

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
        for dim in range(pred_states.shape[1]):
            if dim in dim_skips:
                continue
        
            curr_sequence = pred_states[:,dim:dim+1]
            curr_min = minmax_data[dim,0]
            curr_max = minmax_data[dim,1]

            pred_states[:,dim:dim+1] = (curr_sequence - curr_min) / (curr_max - curr_min)
        
        return pred_states
    
    ###### NESTED FUNCTIONS END ######

    # Parsing the navigation states
    Xdot_prev = input_unnorm[:, 0]  # Position in X (batch_size,)
    Ydot_prev = input_unnorm[:, 1]  # Position in Y (batch_size,)
    q_prev = input_unnorm[:, 2:4]  # Orientation (batch_size,2)
    omega_prev = input_unnorm[:, 4]  # Angular velocity (batch_size,)
    yaw_batch = quat2theta(q_prev.clone())  # Convert quaternion to yaw (batch_size,)

    #print(f'Xdot_prev: {torch.isnan(Xdot_prev).any()}, Ydot_prev: {torch.isnan(Ydot_prev).any()}, q_prev: {torch.isnan(q_prev).any()}, omega_prev: {torch.isnan(omega_prev).any()}, yaw_batch: {torch.isnan(yaw_batch).any()}')
    # Compute the Jacobian for the batch of theta values
    J_batch = jacobian_batch(yaw_batch)

    #print(f'input_unnorm (first ts): {input_unnorm[:,0,:]}')
    #print(f'input_unnorm (second ts): {input_unnorm[:,1,:]}')

    # Compute the linear and angular velocities for each robot in the batch
    uu_batch = input_unnorm[:, 6:8]
    #print(f'uu_batch: {uu_batch}')
    vel_batch = torch.bmm(J_batch, uu_batch.unsqueeze(2)).squeeze(-1)  # Batched matrix multiplication
    Xdot_next = vel_batch[:, 0]  # Linear velocity in X (batch_size,)
    Ydot_next = vel_batch[:, 1]  # Linear velocity in Y (batch_size,)
    omega_next = vel_batch[:, 2]  # Angular velocity (batch_size,)

   # print(f'Xdot_next: {Xdot_next}, Ydot_next: {Ydot_next}, omega_next: {omega_next}')

    yaw_batch += omega_next * dt  # Update the orientation
    q_next = theta2quat(yaw_batch)  # Convert the updated yaw to quaternion

    # Forming predictive states for NP decoder
    pred_states = torch.cat([Ydot_next.unsqueeze(1), Xdot_next.unsqueeze(1), q_next[:,:2], omega_next.unsqueeze(1)], dim=1)

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
def train_dkf_model(model, replay_buffer, optimizer, loss_function, batch_size=int(5e3)):
    # Setting the model to training mode
    model.train()

    # Randomly sample a batch of data from the replay buffer
    batch = [replay_buffer[i] for i in torch.randint(0, len(replay_buffer), (batch_size,))]

    # Separate the batch into low and high datasets
    batch_low = torch.stack([item[0] for item in batch]).squeeze(1)  # Extract sample_low
    batch_high = torch.stack([item[1] for item in batch]).squeeze(1)  # Extract sample_high

    # Normalizing batch input tensor for the NP model    
    norm_batch_low = minmax_norm(batch_low.clone())
    norm_batch_high = minmax_norm(batch_high.clone())

    pred_states = compute_state_priors(batch_low, dt=0.02)

    prev_x_states = norm_batch_low[:,:5]
    prev_z_states = norm_batch_high[:,:5]
    next_x_states = norm_batch_low[:,8:]
    next_z_states = norm_batch_high[:,8:]
    dt_array = norm_batch_low[:,5:6]
    uu_array = norm_batch_low[:,6:8]
    batch_latent_transitions = torch.cat([prev_z_states, dt_array, uu_array, next_z_states], dim=1)
    batch_observed_transitions = torch.cat([prev_x_states, dt_array, uu_array, pred_states], dim=1)

    # forward + loss computation
    z_pred, P_z, x_pred, R_x = model(batch_latent_transitions, batch_observed_transitions)
    loss = loss_function(z_pred, next_z_states, x_pred, next_x_states)

    # Backpropagate and optimize
    optimizer.zero_grad()  # Clear gradients from previous step
    loss.backward()  # Compute gradients
    optimizer.step()  # Update the model weights

    # Compute batch average absolute error (AAE)
    z_pred, P_z, next_z_states = minmax_unnorm(z_pred, P_z, next_z_states)
    RMSE, RMSE_std = compute_RMSE(next_z_states, z_pred)
    NLL = compute_avg_nll(next_z_states, z_pred, P_z)

    return model, loss, RMSE, RMSE_std, NLL