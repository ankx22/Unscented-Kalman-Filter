# from rotplot import rotplot
from PIL import Image
import os
import sys
import time
import math
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy import io
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from helperfunctions import *

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


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
    init_quat = angle_to_quat(initial_orientation)
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
        fused_quaternion = normalize(fused_quaternion)

        resultant_quaternions[0, i+1] = np.float64(fused_quaternion[0])
        resultant_quaternions[1, i+1] = np.float64(fused_quaternion[1])
        resultant_quaternions[2, i+1] = np.float64(fused_quaternion[2])
        resultant_quaternions[3, i+1] = np.float64(fused_quaternion[3])
        current_quaternion = fused_quaternion/np.linalg.norm(fused_quaternion)

    return resultant_quaternions


def process_update(x_prevt, gyro, delta_t, P, Q):
    n = 6
    S = np.linalg.cholesky(P + Q)
    W = np.hstack((np.sqrt(n)*S, -np.sqrt(n)*S))
    # print(W.shape)
    sigma_points = np.zeros((7, 12))
    tf_sigma_points = np.zeros((7, 12))
    X_qi = np.zeros((4, 2*n))
    X_wi = W[3:6, :] + x_prevt[4:7].reshape(-1, 1)
    Y_qi = np.zeros((4, 2*n))
    for i in range(2*n):
        q_i = qfromRV(W[:, i])
        X_qi[:, i] = quat_multiply(x_prevt[0:4], q_i)
        sigma_points[:, i] = np.vstack(
            (X_qi[:, i].reshape(-1, 1), X_wi[:, i].reshape(-1, 1))).flatten()
        w = sigma_points[4:7, i]
        delta_q = qfromRV(w, delta_t)
        Y_qi[:,i] = quat_multiply(sigma_points[0:4,i],delta_q)
    Y_wi = sigma_points[4:7, :]
    tf_sigma_points = np.vstack((Y_qi,Y_wi))

    mu_bar = intrinsicGradientDescent(tf_sigma_points, sigma_points[:,0])
    q_bar_inv = quaternion_inverse(mu_bar[0:4])
    omega_bar = mu_bar[4:7]
    W_prime = []
    P_bar = np.zeros((6, 6))
    for i in range(len(tf_sigma_points)):
        quat_comp = quat_multiply(tf_sigma_points[i][0:4], q_bar_inv)
        omega_comp = tf_sigma_points[i][4:7] - omega_bar
        # Wi_prime = np.concatenate([quat_comp,omega_comp])
        r = R.from_quat(quat_comp.reshape(4,))
        yaw, pitch, roll = r.as_euler('zyx', degrees=False)
        euler_array = np.array([roll, pitch, yaw]).reshape(-1, 1)
        Wi_prime_euler = np.concatenate([euler_array, omega_comp])
        W_prime.append(Wi_prime_euler)
        P_bar += np.dot(Wi_prime_euler, np.transpose(Wi_prime_euler))
    P_bar = P_bar/2*n

    return W_prime, tf_sigma_points, mu_bar, P_bar


def measurement_update(W_prime, Y, R, P_prev, x_prev, z):
    g_quat = np.array([0, 0, 0, 1]).reshape(-1, 1)
    print(f"g shape: {g_quat[0,3]}")
    z_bar = []
    z_bar_euler = []
    z_bar_mean = np.zeros((7, 1))
    for i in range(len(Y)):
        Yq = Y[i][0:4]
        Yw = Y[i][4:7]
        z_bar_quat = quat_multiply(quat_multiply(
            quaternion_inverse(Yq), g_quat), Yq)
        z_bar.append(np.concatenate([z_bar_quat, Yw]))
        z_bar_mean += np.concatenate([z_bar_quat, Yw])
        z_bar_mean_i = vec_from_state(z_bar[i])
        z_bar_euler.append(z_bar_mean_i)
    z_bar_mean = z_bar_mean/len(Y)
    z_bar_mean_euler = vec_from_state(z_bar_mean)
    P_zz = np.zeros((6, 6))
    P_xz = np.zeros((6, 6))
    for i in range(len(Y)):
        component1 = z_bar_mean_euler - z_bar_euler[i]
        P_zz += np.dot(component1, component1.T)
        P_xz += np.dot(W_prime[i], component1.T)
    P_zz = P_zz/len(Y)
    P_vv = P_zz + R
    P_xz = P_xz/len(Y)
    K = np.dot(P_xz, np.linalg.inv(P_vv))
    P = P_prev - np.dot(np.dot(K, P_vv), K.T)
    update_term = np.dot(K, (z - z_bar_mean_euler))
    update_term_quat = np.concatenate(
        [angle_to_quat(update_term[0:3]), update_term[3:6]])
    x = x_prev + update_term_quat
    return x, P


def ukf(acc, gyro, vicon_rpy, timestamps):
    # initial_orientation = np.array([np.average(vicon_rpy[0, 0:200]), np.average(
    #     vicon_rpy[1, 0:200]), np.average(vicon_rpy[2, 0:200])])
    initial_orientation = np.array([1, 0, 0, 0, 0, 0, 0], dtype=np.float64)
    _, N = acc.shape  # Get the size of IMU data
    # init_quat = angle_to_quat(initial_orientation).reshape(-1, 1)
    # init_quat = normalize(init_quat)
    z_stacked = np.vstack((acc, gyro))
    P = 1e-2 * np.eye(6)
    Q = np.diag([100, 100, 100, 0.1, 0.1, 0.1])
    R = np.diag([0.5, 0.5, 0.5, 0.01, 0.01, 0.01])
    # initial state vector
    # x_t = np.vstack((initial_orientation[0:4], initial_orientation[4:7]))
    x_t = initial_orientation.T
    ukf_quats = np.zeros((4, N))
    ukf_quats[:, 0] = initial_orientation[0:4]
    for i in range(N-1):
        if i == 0:
            delta_t = 0.01
        else:
            delta_t = timestamps[i+1] - timestamps[i]
        # Function for process update
        Wi, Yi, P_bar = process_update(x_t, gyro[:, i], delta_t, P, Q)
        # Function for measurement update
        # x_t, P = measurement_update(Wi, Yi, mu_bar, P_bar, R, acc)
        x_t, P = measurement_update(
            Wi, Yi, R, P, x_t, z_stacked[:, i].reshape(-1, 1))
        ukf_quats[:, i] = normalize(x_t[0:4].reshape(4,))

    orientations = np.zeros((3, N))
    for i in range(N):
        qw = ukf_quats[0, i]
        qx = ukf_quats[1, i]
        qy = ukf_quats[2, i]
        qz = ukf_quats[3, i]
        # r = R.from_quat([qw,qx,qy,qz])
        # yaw, pitch, roll = r.as_euler('zyx', degrees=False)
        roll = np.arctan2(2*(qw*qx + qy*qz), 1-2*(qx*2 + qy*2))
        sinp = 2*(qw*qy-qx*qz)
        if abs(sinp) >= 1:
            # Use 90 degrees if out of range
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)

        yaw = np.arctan2(2*(qw*qz + qx*qy), 1-2*(qy*2+qz*2))

        # orientations[0,i] = np.arctan2(2*(qw*qx + qy*qz),1-2*(qx*2 + qy*2))
        # orientations[1,i] = np.arcsin(2*(qw*qy-qx*qz))
        # orientations[2,i] = np.arctan2(2*(qw*qz + qx*qy),1-2*(qy*2+qz*2))
        orientations[0, i] = roll
        orientations[1, i] = pitch
        orientations[2, i] = yaw
    return orientations


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
    IMU_filename = "imuRaw1"
    vicon_filename = "viconRot1"
    # Loading the IMU data, parameters and Vicon Groundtruth data
    absolute_path = os.path.dirname(__file__)

    # Use this for train data
    relativepath_IMUdata = "Data/Train/IMU/"+IMU_filename+".mat"
    fullpath_IMUdata = os.path.join(absolute_path, '..', relativepath_IMUdata)
    relativepath_IMUparams = 'IMUParams.mat'
    fullpath_IMUparams = os.path.join(
        absolute_path, '..', relativepath_IMUparams)

    relativepath_vicon = 'Data/Train/Vicon/'+vicon_filename+'.mat'
    fullpath_vicon = os.path.join(absolute_path, '..', relativepath_vicon)

    # Use this for test data
    # relativepath_IMUdata = "Data/Test/IMU/"+IMU_filename+".mat"
    # fullpath_IMUdata = os.path.join(absolute_path, '..', relativepath_IMUdata)
    # relativepath_IMUparams = 'IMUParams.mat'
    # fullpath_IMUparams = os.path.join(
    #     absolute_path, '..', relativepath_IMUparams)

    IMU_data = io.loadmat(fullpath_IMUdata)
    IMU_params = io.loadmat(fullpath_IMUparams)['IMUParams']
    vicon_data = io.loadmat(fullpath_vicon)

    # Seperating the timestamps and values for IMU and Vicon data
    # Each column represents the vector of six values (along the rows)
    IMU_vals = IMU_data['vals']
    IMU_ts = IMU_data['ts']
    vicon_rotmat = vicon_data['rots']  # ZYX Euler angles rotation matrix.
    vicon_ts = vicon_data['ts']

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
    vicon_rpy = mat_to_angle(vicon_rotmat)
    # vicon_rpy = np.array(([0, 0, 0]))

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
    # Roll, Pitch and yaw using ukf filter
    ukf_orientation = ukf(
        IMU_vals_converted[0:3, :], omega_xyz, vicon_rpy, IMU_ts[0, :])
    # Plotting the orientations obtained from different methods
    fig = plt.figure(figsize=(10, 10))

    ax1 = fig.add_subplot()
    ax1 = plt.subplot(3, 1, 1, title='Roll (X)')
    ax1.plot(vicon_ts[0], vicon_rpy[0, :], label='vicon')
    # ax1.plot(IMU_ts[0], gyro_orientaion[0, :], label='gyro')
    # ax1.plot(IMU_ts[0], acc_orientation[0, :], label='acc')
    # ax1.plot(IMU_ts[0], comp_orientation[0, :], label='comp')
    # ax1.plot(IMU_ts[0], madgwick_orientation[0, :], label='madg')
    ax1.plot(IMU_ts[0], ukf_orientation[0, :], label='ukf')
    plt.xlabel("timesteps")
    plt.ylabel("angles (rad)")
    plt.legend()
    ax2 = fig.add_subplot()
    ax2 = plt.subplot(3, 1, 2, title='Pitch (Y)')
    ax2.plot(vicon_ts[0], vicon_rpy[1, :], label='vicon')
    # ax2.plot(IMU_ts[0], gyro_orientaion[1, :], label='gyro')
    # ax2.plot(IMU_ts[0], acc_orientation[1, :], label='acc')
    # ax2.plot(IMU_ts[0], comp_orientation[1, :], label='comp')
    # ax2.plot(IMU_ts[0], madgwick_orientation[1, :], label='madg')
    ax2.plot(IMU_ts[0], ukf_orientation[1, :], label='ukf')
    plt.xlabel("timesteps")
    plt.ylabel("angles (rad)")
    plt.legend()
    ax3 = fig.add_subplot()
    ax3 = plt.subplot(3, 1, 3, title='Yaw (Z)')
    ax3.plot(vicon_ts[0], vicon_rpy[2, :], label='vicon')
    # ax3.plot(IMU_ts[0], gyro_orientaion[2, :], label='gyro')
    # ax3.plot(IMU_ts[0], acc_orientation[2, :], label='acc')
    # ax3.plot(IMU_ts[0], comp_orientation[2, :], label='comp')
    # ax3.plot(IMU_ts[0], madgwick_orientation[2, :], label='madg')
    ax3.plot(IMU_ts[0], ukf_orientation[2, :], label='ukf')
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
