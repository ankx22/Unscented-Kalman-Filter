from rotplot import rotplot
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from scipy import io
import numpy as np
import time
import sys
import os
from PIL import Image


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def mat_to_angle(vicon_data):
    """
    Converts the rotation matrices (ZYX) from vicon data to roll pitch and yaw.

    Args:
        vicon_data : (3,3,N) array of vicon groundtruth data.

    Returns:
        orientations: (3,N) array of roll, pitch, and yaw angles (in radians)
    """
    # Create an empty orientations array
    _, _, N = vicon_data.shape
    orientations = np.zeros((3, N))
    # Convert ZYX mat to rpy angles
    for i in range(N):
        matrix = vicon_data[:, :, i]
        r11, r12, r13 = matrix[0]
        r21, r22, r23 = matrix[1]
        r31, r32, r33 = matrix[2]
        if r11 and r21 == 0:
            orientations[:, i] = [np.arctan2(r12, r22), np.pi/2, 0]
        else:
            orientations[:, i] = [np.arctan2(
                r32, r33), np.arctan2(-r31, np.sqrt(r11**2 + r21**2)), np.arctan2(r21, r11)]
    return (orientations)


def quat_multiply(q1, q2):
    """
    Multiply two quaternions.

    Parameters:
        q1 (numpy.ndarray): First quaternion as a 4-element array [w, x, y, z].
        q2 (numpy.ndarray): Second quaternion as a 4-element array [w, x, y, z].

    Returns:
        numpy.ndarray: Resultant quaternion after multiplication.
    """
    a1, a2, a3, a4 = q1[0], q1[1], q1[2], q1[3]
    b1, b2, b3, b4 = q2[0], q2[1], q2[2], q2[3]
    return np.transpose(np.array([a1*b1 - a2*b2 - a3*b3 - a4*b4, a1*b2 + a2*b1 + a3*b4 - a4*b3, a1*b3 - a2*b4 + a3*b1 + a4*b2, a1*b4 + a2*b3 - a3*b2 + a4*b1]))


def to_quaternion(rpy_orientation):
    """
    Convert roll-pitch-yaw (Euler angles) to quaternion.

    Parameters:
        rpy_orientation (numpy.ndarray): Input Euler angles as a 3-element array [roll, pitch, yaw].

    Returns:
        numpy.ndarray: Corresponding quaternion as a 4-element array [w, x, y, z].
    """
    # roll (x), pitch (Y), yaw (z)

    roll = rpy_orientation[0]
    pitch = rpy_orientation[1]
    yaw = rpy_orientation[2]
    # Abbreviations for the various angular functions
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)

    q = []
    q.append(cr * cp * cy + sr * sp * sy)
    q.append(sr * cp * cy - cr * sp * sy)
    q.append(cr * sp * cy + sr * cp * sy)
    q.append(cr * cp * sy - sr * sp * cy)

    return np.transpose(np.array(q))


def angle_to_mat(angles):
    """
    Converts the Euler angles to Rotation matrix (ZYX).

    Args:
        angles : (3,N) array of roll, pitch, and yaw.

    Returns:
        matrices: (3,3,N) array of rotation matrices.
    """
    _, N = angles.shape
    matrices = np.zeros((3, 3, N))

    for i in range(N):
        roll = angles[0, i]
        pitch = angles[1, i]
        yaw = angles[2, i]
        mat1 = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                        [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        mat2 = np.array([[np.cos(pitch), 0, np.sin(pitch)], [
                        0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
        mat3 = np.array(
            [[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
        matrices[:, :, i] = np.dot(np.dot(mat1, mat2), mat3)
    return matrices


def only_gyro(omega, initial_o, timestamps):
    """
    Performs numerical integration of the IMU gyroscope values and returns the orientations.

    Args:
        omega : (3,N) array of angular velocities (in [omega_x, omega_y, omega_z] format each).
        initial_o :(1,3) array of average of the first 200 orientations obtained from vicon ground truth.
        timestamps: (1,N) array of the corresponding IMU time stamps.

    Returns:
        orientations: (3,N) array of roll, pitch, and yaw angles (in radians)
    """
    # Creating an empty orientations array
    _, N = omega.shape
    orientations = np.zeros((3, N))
    orientations[:, 0] = initial_o
    # Numerical integration
    for i in range(N-1):
        # roll (phi), pitch(theta), and yaw (psi)
        phi = orientations[0, i]
        theta = orientations[1, i]
        psi = orientations[2, i]
        # conversion matrix to convert angular velocities (omegas) to rate of change of roll,pitch, and yaw.
        conv_mat = np.array([[1, np.sin(phi)*np.tan(theta), np.cos(phi)*np.tan(theta)], [
                            0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi)/np.cos(theta), np.cos(phi)/np.cos(theta)]])
        # rpy_{t+1} = rpy_{t} + rpy_dot*dt
        orientations[:, i+1] = orientations[:, i] + \
            np.dot(conv_mat, omega[:, i])*(timestamps[0, i+1]-timestamps[0, i])

    return (orientations)


def normalize_quaternion(q):
    """
    Normalize a quaternion to have a unit magnitude.

    Parameters:
        q (numpy.ndarray): Input quaternion as a 4-element array [w, x, y, z].

    Returns:
        numpy.ndarray: Normalized quaternion with unit magnitude.
    """
    magnitude = np.linalg.norm(q)
    normalized_q = q / magnitude
    return normalized_q


def quaternion_to_euler(q):
    """
    Convert a quaternion to roll-pitch-yaw (Euler angles) using ZYX rotation order.

    Parameters:
        q (numpy.ndarray): Input quaternion as a 4-element array [w, x, y, z].

    Returns:
        tuple: Tuple containing roll, pitch, and yaw angles (in radians) in ZYX rotation order.
    """
    # Extract quaternion components
    w, x, y, z = q

    # ZYX rotation order (yaw-pitch-roll)
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y**2 + z**2))
    pitch = np.arcsin(2*(w*y - z*x))
    roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x**2 + y**2))

    return roll, pitch, yaw


def only_acc(acc):
    """
    Returns orientations from IMU acceleration data assuming gravity is pointing downwards (-Z).

    Args:
        acc : (3,N) array of accelerations (in [a_x, a_y, a_z] format each).

    Returns:
        orientations: (3,N) array of roll, pitch, and yaw angles (in radians)
    """
    # Creating an empty orientations array
    _, N = acc.shape
    orientations = np.zeros((3, N))
    for i in range(N):
        orientations[:, i] = [np.arctan2(acc[1, i], np.sqrt(acc[0, i]**2 + acc[2, i]**2)),
                              np.arctan2(-acc[0, i],
                                         np.sqrt(acc[1, i]**2 + acc[2, i]**2)),
                              np.arctan2(np.sqrt(acc[0, i]**2 + acc[1, i]**2), acc[2, i])]

    return (orientations)


def comp_filter(acc, gyro, vicon_rpy, timestamps):
    """
    Returns orientations using complementary filter.

    Args:
        acc : (3,N) array of accelerations (in [a_x, a_y, a_z] format each).
        omega : (3,N) array of angular velocities (in [omega_x, omega_y, omega_z] format each).
        vicon_rpy : (3,N) array of ground truth vicon roll, pitch, and yaw angles.
        timestamps : (1,N) array of the corresponding IMU time stamps.


    Returns:
        orientations: (3,N) array of roll, pitch, and yaw angles (in radians)
    """
    _, N = acc.shape  # Get the size of IMU data
    # Filtering the data first
    # Running a low pass filter on acceleration
    n = 0.8
    acc_filtered = np.zeros((3, N))
    acc_filtered[:, 0] = acc[:, 0]
    for i in range(N-1):
        acc_filtered[:, i+1] = (1-n)*acc[:, i+1] + n*acc_filtered[:, i]

    # Running a high pass filter on gyroscope data
    gyro_filtered = np.zeros((3, N))
    gyro_filtered[:, 0] = gyro[:, 0]
    for i in range(N-1):
        gyro_filtered[:, i+1] = (1-n)*gyro_filtered[:, i] + \
            (1-n)*(gyro[:, i+1]-gyro[:, i])

    # Getting orientations from the filtered data
    # initial_orientation = np.array([np.average(vicon_rpy[0, 0:200]), np.average(
    #     vicon_rpy[1, 0:200]), np.average(vicon_rpy[2, 0:200])])
    initial_orientation = np.array([0, 0, 0])
    gyro_orientations = only_gyro(
        gyro_filtered, initial_orientation, timestamps)
    acc_orientations = only_acc(acc_filtered)
    # Fusing the orientations with high and low pass filters
    alpha = 0.8
    beta = 0.8
    gamma = 0.9
    # orientations = (1-alpha)*np.array(gyro_orientations) + alpha*np.array(acc_orientations)
    orientations = np.dot(np.array([[1-alpha, 0, 0], [0, 1-beta, 0], [0, 0, 1-gamma]]), np.array(
        gyro_orientations)) + np.dot(np.array([[alpha, 0, 0], [0, beta, 0], [0, 0, gamma]]), np.array(acc_orientations))
    return orientations


def madgwick(acc, gyro, vicon_rpy, timestamps):
    """
    Returns orientations using madgwick filter.

    Args:
        acc : (3,N) array of accelerations (in [a_x, a_y, a_z] format each).
        omega : (3,N) array of angular velocities (in [omega_x, omega_y, omega_z] format each).
        vicon_rpy : (3,N) array of ground truth vicon roll, pitch, and yaw angles.
        timestamps : (1,N) array of the corresponding IMU time stamps.


    Returns:
        orientations: (4,N) array of quaternions
    """
    # Getting orientations from the filtered data
    # initial_orientation = np.array([np.average(vicon_rpy[0, 0:200]), np.average(
    #     vicon_rpy[1, 0:200]), np.average(vicon_rpy[2, 0:200])])
    initial_orientation = np.array([0, 0, 0])
    _, N = acc.shape  # Get the size of IMU data

    # Getting dt timesteps
    time_differences = []
    for i in range(1, len(timestamps[0])):
        time_diff = timestamps[0][i] - timestamps[0][i - 1]
        time_differences.append(time_diff)

    # get init orientation in quat (4x1)
    init_quat = to_quaternion(initial_orientation)
    beta = 0.1
    gyro_data = np.vstack((np.zeros(N), gyro))
    acc_data = np.vstack((np.zeros(N), acc))
    resultant_quaternions = np.zeros((4, N))
    current_quaternion = init_quat/np.linalg.norm(init_quat)
    resultant_quaternions[:, 0] = current_quaternion
    for i in range(N-1):
        q_incremental_gyro = 1/2 * \
            quat_multiply(current_quaternion, gyro_data[:, i+1])

        q1 = np.float64(current_quaternion[0])
        q2 = np.float64(current_quaternion[1])
        q3 = np.float64(current_quaternion[2])
        q4 = np.float64(current_quaternion[3])

        # print(type(current_quat[0]))
        jacobian_acceleration = np.array([[-2*q3, 2*q4, -2*q1, 2*q2],
                                          [2*q2, 2*q1, 2*q4, 2*q3],
                                          [0, -4*q2, -4*q3, 0]
                                          ])

        f_acc = np.array([[2*(q2*q4-q1*q3) - acc_data[1, i+1]],
                          [2*(q1*q2 + q3*q4) - acc_data[2, i+1]],
                          [2*(0.5 - q2*q2 - q3*q3) - acc_data[3, i+1]]])

        gradient_f_acc = jacobian_acceleration.T.dot(f_acc)
        norm_grad_f_acc = np.linalg.norm(gradient_f_acc)
        q_incremental_acc = -beta * (gradient_f_acc/norm_grad_f_acc)
        q_incremental_gyro = q_incremental_gyro.reshape(4, 1)
        quaternion_increment_fused = q_incremental_gyro + q_incremental_acc
        current_quaternion = current_quaternion.reshape(4, 1)
        fused_quaternion = current_quaternion + \
            time_differences[i] * quaternion_increment_fused
        fused_quaternion = normalize_quaternion(fused_quaternion)

        resultant_quaternions[0, i+1] = np.float64(fused_quaternion[0])
        resultant_quaternions[1, i+1] = np.float64(fused_quaternion[1])
        resultant_quaternions[2, i+1] = np.float64(fused_quaternion[2])
        resultant_quaternions[3, i+1] = np.float64(fused_quaternion[3])
        current_quaternion = fused_quaternion/np.linalg.norm(fused_quaternion)

    return resultant_quaternions


def normalize(v):
    norm=np.linalg.norm(v)
    if norm==0:
        norm=np.finfo(v.dtype).eps
    return v/norm


def quaternion_inverse(q):
    # Calculate the magnitude squared of the quaternion
    magnitude_squared = np.linalg.norm(q)**2
    
    # Check if the quaternion is close to zero
    if magnitude_squared < 1e-10:
        raise ValueError("Cannot compute the inverse of a zero quaternion")
    
    # Calculate the conjugate of the quaternion
    conjugate = np.array([q[0], -q[1], -q[2], -q[3]])
    
    # Calculate the inverse
    inverse = conjugate / magnitude_squared
    
    return inverse


def intrinsicGradientDescent(tf_sigma_points):
    """
    Data: tf_sigma_points
    result: mean of tf_sigma_points
    """

    qt_list = []
    for i in range(len(tf_sigma_points)):
        qt_list.append(tf_sigma_points[i][0:4,0])
    qt = tf_sigma_points[0][0:4,0]
    state_sum = sum(tf_sigma_points)
    omega_bar = state_sum[4:7]/len(tf_sigma_points)
    max_iter = 100 #tunable 
    error_threshold = 1e-4 #tunable
    current_iter = 0 
    mean_quat_err = float('inf')
    while current_iter < max_iter:
        q_inv = quaternion_inverse(qt)
        total_quat_err = np.zeros_like(qt)
        for quat in qt_list:
            err_quat = quat_multiply(quat,q_inv)    
            total_quat_err += err_quat  
        mean_quat_err = total_quat_err/len(tf_sigma_points)
        qt = quat_multiply(mean_quat_err,qt)
        current_iter += 1 
        if current_iter == max_iter or np.linalg.norm(mean_quat_err) < error_threshold:
            q_bar = qt 
    return np.concatenate([q_bar, omega_bar])



def process_update(x_prevt, gyro, delta_t, P, Q):
    n = 6
    S = np.linalg.cholesky(P + Q)
    W = np.hstack((-np.sqrt(n)*S, np.sqrt(n)*S))
    sigma_points = []
    tf_sigma_points = []
    for i in range(2*n):
        quat_vec = np.sin(
            0.5*np.linalg.norm(W[0:3, i])*delta_t)*normalize(W[0:3,i])
        qwi = np.transpose(np.array([np.cos(
            0.5*np.linalg.norm(W[0:3, i])*delta_t), quat_vec[0], quat_vec[1], quat_vec[2]]))
        omega_wi = np.transpose(np.array([W[3, i], W[4, i], W[5, i]]))
        sigma_points.append(np.vstack((quat_multiply(
            x_prevt[0:4, 0], qwi).reshape(-1, 1), (x_prevt[4:7, 0]+omega_wi).reshape(-1, 1))))
        quat_angle = np.sin(
            0.5*np.linalg.norm(x_prevt[4:7, 0])*delta_t)*normalize(x_prevt[4:7, 0])
        q_delta = np.transpose(np.array([np.cos(
            0.5*np.linalg.norm(x_prevt[4:7, 0])*delta_t), quat_angle[0], quat_angle[1], quat_angle[2]]))
        tf_sigma_points.append(np.vstack((quat_multiply((quat_multiply(
            x_prevt[0:4, 0], qwi).reshape(-1, 1)), q_delta).reshape(-1, 1), (x_prevt[4:7, 0]+omega_wi).reshape(-1, 1))))
        
    mu_bar = intrinsicGradientDescent(tf_sigma_points)
    q_bar_inv = quaternion_inverse(mu_bar[0:4])
    omega_bar = mu_bar[4:7]
    W_prime = []
    P_bar = np.zeros(7,7)
    for i in range(len(tf_sigma_points)):
        quat_comp = quat_multiply(tf_sigma_points[i][0:4],q_bar_inv)
        omega_comp = tf_sigma_points[i][4:7] - omega_bar
        Wi_prime = np.concatenate([quat_comp,omega_comp])
        W_prime.append(Wi_prime)
        P_bar += np.linalg.norm(Wi_prime)**2
    P_bar = P_bar/2*n 




def ukf(acc, gyro, vicon_rpy, timestamps):
    initial_orientation = np.array([np.average(vicon_rpy[0, 0:200]), np.average(
        vicon_rpy[1, 0:200]), np.average(vicon_rpy[2, 0:200])])
    _, N = acc.shape  # Get the size of IMU data
    init_quat = to_quaternion(initial_orientation)
    P = np.zeros((6, 6))
    Q = np.diag([100, 100, 100, 1, 1, 1])
    R = np.diag([10, 10, 10, 1, 1, 1])
    # initial state vector
    x_t = np.vstack((init_quat, gyro[:, 0].reshape(-1,1)))
    for i in range(N-1):
        delta_t = timestamps[i+1] - timestamps[i]
        # Function for process update
        Wi, Yi, mu_bar, P_bar = process_update(x_t, gyro, delta_t, P, Q)
        # Function for measurement update
        x_t, P = measurement_update(Wi, Yi, mu_bar, P_bar, R, acc)


def animate(i, acc_orientations, gyro_orientations, comp_orientations, madg_orientations, vicon_mats):
    """
    This function helps in creation of the videos.

    """
    acc_mats = angle_to_mat(acc_orientations)
    gyro_mats = angle_to_mat(gyro_orientations)
    comp_mats = angle_to_mat(comp_orientations)
    madg_mats = angle_to_mat(madg_orientations)

    ax1 = plt.subplot(151, projection='3d',
                      title='gyro (i ='+str(i)+')', adjustable='datalim')
    ax2 = plt.subplot(152, projection='3d',
                      title='acc(i ='+str(i)+')', adjustable='datalim')
    ax3 = plt.subplot(153, projection='3d',
                      title='CF(i ='+str(i)+')', adjustable='datalim')
    ax4 = plt.subplot(154, projection='3d',
                      title='madg(i ='+str(i)+')', adjustable='datalim')
    ax5 = plt.subplot(155, projection='3d',
                      title='vicon(i ='+str(i)+')', adjustable='datalim')

    rotplot(gyro_mats[:, :, 10*i], ax1)
    rotplot(acc_mats[:, :, 10*i], ax2)
    rotplot(comp_mats[:, :, 10*i], ax3)
    rotplot(madg_mats[:, :, 10*i], ax4)
    rotplot(vicon_mats[:, :, 10*i], ax5)


def main():
    IMU_filename = "imuRaw7"
    vicon_filename = "viconRot1"
    # Loading the IMU data, parameters and Vicon Groundtruth data
    absolute_path = os.path.dirname(__file__)

    # #Use this for train data
    # relativepath_IMUdata = "Data/Train/IMU/"+IMU_filename+".mat"
    # fullpath_IMUdata = os.path.join(absolute_path, '..', relativepath_IMUdata)
    # relativepath_IMUparams = 'IMUParams.mat'
    # fullpath_IMUparams = os.path.join(
    #     absolute_path, '..', relativepath_IMUparams)

    # relativepath_vicon = 'Data/Train/Vicon/'+vicon_filename+'.mat'
    # fullpath_vicon = os.path.join(absolute_path, '..', relativepath_vicon)

    # Use this for test data
    relativepath_IMUdata = "Data/Test/IMU/"+IMU_filename+".mat"
    fullpath_IMUdata = os.path.join(absolute_path, '..', relativepath_IMUdata)
    relativepath_IMUparams = 'IMUParams.mat'
    fullpath_IMUparams = os.path.join(
        absolute_path, '..', relativepath_IMUparams)

    IMU_data = io.loadmat(fullpath_IMUdata)
    IMU_params = io.loadmat(fullpath_IMUparams)['IMUParams']
    # vicon_data = io.loadmat(fullpath_vicon)

    # Seperating the timestamps and values for IMU and Vicon data
    # Each column represents the vector of six values (along the rows)
    IMU_vals = IMU_data['vals']
    IMU_ts = IMU_data['ts']
    # vicon_rotmat = vicon_data['rots']  # ZYX Euler angles rotation matrix.
    # vicon_ts = vicon_data['ts']

    # Converting the data to physical values with units
    # The bias of the gyroscope values is taken as the average of the first 200 gyroscope readings
    bg = np.array([np.average(IMU_vals[3, 0:200]), np.average(
        IMU_vals[4, 0:200]), np.average(IMU_vals[5, 0:200])])  # Gyroscope bias
    # For acceleration values: a_conv = (scale*(a_old) + bias)*9.81
    # For angular velocities: omega_conv = (3300/1023)*(pi/180)*(0.3)*(omega_old - bias_gyro)
    IMU_vals_converted = np.array([(IMU_vals[0, :]*IMU_params[0, 0]+IMU_params[1, 0])*9.81,
                                   (IMU_vals[1, :]*IMU_params[0, 1] +
                                    IMU_params[1, 1])*9.81,
                                   (IMU_vals[2, :]*IMU_params[0, 2] +
                                    IMU_params[1, 2])*9.81,
                                   (3300/1023)*(np.pi/180) *
                                   0.3*(IMU_vals[3, :]-bg[0]),
                                   (3300/1023)*(np.pi/180) *
                                   0.3*(IMU_vals[4, :]-bg[1]),
                                   (3300/1023)*(np.pi/180)*0.3*(IMU_vals[5, :]-bg[2])])

    # Getting the orientations using different methods
    # Converting rotation matrices to roll,pitch,yaw
    # vicon_rpy = mat_to_angle(vicon_rotmat)
    vicon_rpy = np.array(([0, 0, 0]))

    # only gyroscope
    # initial_orientation = np.array([np.average(vicon_rpy[0, 0:200]), np.average(
    #     vicon_rpy[1, 0:200]), np.average(vicon_rpy[2, 0:200])])
    initial_orientation = np.array([0, 0, 0])
    omega_zxy = IMU_vals_converted[3:6, :]
    omega_xyz = np.array([omega_zxy[1, :], omega_zxy[2, :], omega_zxy[0, :]])
    gyro_orientaion = only_gyro(omega_xyz, initial_orientation, IMU_ts)

    # only acceleration
    acc_orientation = only_acc(IMU_vals_converted[0:3, :])

    # Complimentary Filter
    comp_orientation = comp_filter(
        IMU_vals_converted[0:3, :], omega_xyz, vicon_rpy, IMU_ts)

    # getting madgwick filter resultant quaternions
    q_vec = madgwick(IMU_vals_converted[0:3, :],
                     omega_xyz, vicon_rpy, IMU_ts)

    # converting them to roll, pitch and yaw
    N = q_vec.shape[1]
    roll_angles = np.zeros(N)
    pitch_angles = np.zeros(N)
    yaw_angles = np.zeros(N)
    for i in range(N):
        q = q_vec[:, i]
        roll, pitch, yaw = quaternion_to_euler(q)
        roll_angles[i] = roll
        pitch_angles[i] = pitch
        yaw_angles[i] = yaw
    # Roll, Pitch and yaw using madgwick filter
    madgwick_orientation = np.vstack((roll_angles, pitch_angles, yaw_angles))
    # Plotting the orientations obtained from different methods
    fig = plt.figure(figsize=(10, 10))

    ax1 = fig.add_subplot()
    ax1 = plt.subplot(3, 1, 1, title='Roll (X)')
    # ax1.plot(vicon_ts[0], vicon_rpy[0, :], label='vicon')
    ax1.plot(IMU_ts[0], gyro_orientaion[0, :], label='gyro')
    ax1.plot(IMU_ts[0], acc_orientation[0, :], label='acc')
    ax1.plot(IMU_ts[0], comp_orientation[0, :], label='comp')
    ax1.plot(IMU_ts[0], madgwick_orientation[0, :], label='madg')
    plt.xlabel("timesteps")
    plt.ylabel("angles (rad)")
    plt.legend()
    ax2 = fig.add_subplot()
    ax2 = plt.subplot(3, 1, 2, title='Pitch (Y)')
    # ax2.plot(vicon_ts[0], vicon_rpy[1, :], label='vicon')
    ax2.plot(IMU_ts[0], gyro_orientaion[1, :], label='gyro')
    ax2.plot(IMU_ts[0], acc_orientation[1, :], label='acc')
    ax2.plot(IMU_ts[0], comp_orientation[1, :], label='comp')
    ax2.plot(IMU_ts[0], madgwick_orientation[1, :], label='madg')
    plt.xlabel("timesteps")
    plt.ylabel("angles (rad)")
    plt.legend()
    ax3 = fig.add_subplot()
    ax3 = plt.subplot(3, 1, 3, title='Yaw (Z)')
    # ax3.plot(vicon_ts[0], vicon_rpy[2, :], label='vicon')
    ax3.plot(IMU_ts[0], gyro_orientaion[2, :], label='gyro')
    ax3.plot(IMU_ts[0], acc_orientation[2, :], label='acc')
    ax3.plot(IMU_ts[0], comp_orientation[2, :], label='comp')
    ax3.plot(IMU_ts[0], madgwick_orientation[2, :], label='madg')
    fig.tight_layout()
    plt.xlabel("timesteps")
    plt.ylabel("angles (rad)")
    plt.legend()
    plt.show()

    # Creating videos (Please uncomment when using)
    # ani = animation.FuncAnimation(plt.gcf(), animate, frames=520, fargs=(
    #     acc_orientation, gyro_orientaion, comp_orientation, madgwick_orientation, vicon_rpy), repeat=False)
    # writervideo = animation.FFMpegWriter(fps=60)
    # ani.save('madgtrain6.mp4', writer=writervideo)


if __name__ == '__main__':
    main()
