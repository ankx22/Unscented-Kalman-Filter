import math
import numpy as np
from scipy.spatial.transform import Rotation as R


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
    return np.array([a1*b1 - a2*b2 - a3*b3 - a4*b4, a1*b2 + a2*b1 + a3*b4 - a4*b3, a1*b3 - a2*b4 + a3*b1 + a4*b2, a1*b4 + a2*b3 - a3*b2 + a4*b1])


def angle_to_quat(rpy_orientation):
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

    return (np.array(q))


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


def quat2RV(q):
    sinalpha = np.linalg.norm(q[1:4])
    cosalpha = q[0]

    alpha = math.atan2(sinalpha, cosalpha)
    if (sinalpha == 0):
        rv = np.array([0, 0, 0], dtype=np.float64)
        return rv

    e = q[1:4]/float(sinalpha)
    rv = e*2.*alpha
    return rv


def qfromRV(eul_angles, dtt=1.0):
    norm = np.linalg.norm(eul_angles)
    dtt = dtt
    if norm == 0:
        q = np.array([1, 0, 0, 0], dtype=np.float64)
        return q
    alpha = norm*dtt
    ew = eul_angles*(dtt/alpha)
    q = np.array([math.cos(alpha/2),
                  ew[0]*math.sin(alpha/2),
                  ew[1]*math.sin(alpha/2),
                  ew[2]*math.sin(alpha/2)])
    q = q/np.linalg.norm(q)
    return q


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        norm = np.finfo(v.dtype).eps
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


def intrinsicGradientDescent(tf_sigma_points, sigma_point):
    """
    Data: tf_sigma_points
    result: mean of tf_sigma_points
    """
    q_bar = sigma_point[0:4]
    current_iter = 0
    max_iter = 100  # tunable
    mean_error = np.array([10000., 10000., 10000.])
    error_threshold = 1e-4  # tunable
    error_vector = np.zeros((3, 12))
    while np.linalg.norm(mean_error) > error_threshold and current_iter < max_iter:
        for i in range(tf_sigma_points.shape[1]):
            q_i = tf_sigma_points[0:4, i]
            error = quat_multiply(q_i, quaternion_inverse(q_bar))
            error_vector[:, i] = quat2RV(error)
        mean_error = np.mean(error_vector, axis=1)
        mean_error_quat = qfromRV(mean_error)
        q_bar = quat_multiply(mean_error_quat, q_bar)
        q_bar = normalize(q_bar)
        current_iter += 1
    omega_bar = (
        1/float(tf_sigma_points.shape[1]))*np.mean(tf_sigma_points[4:7, :], axis=1)
    result = np.vstack(
        (q_bar.reshape(-1, 1), omega_bar.reshape(-1, 1))).reshape(-1,)
    return result
