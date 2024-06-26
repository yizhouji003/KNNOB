import numpy as np
from numpy.linalg import inv, norm
import scipy.integrate
from scipy.spatial.transform import Rotation
import control

def quat_dot(quat, omega):
    """
    Parameters:
        quat, [i,j,k,w]
        omega, angular velocity of body in body axes

    Returns
        quat_dot, [i,j,k,w]

    """
    # Adapted from "Quaternions And Dynamics" by Basile Graf.
    (q0, q1, q2, q3) = (quat[0], quat[1], quat[2], quat[3])
    G = np.array([[q3,  q2, -q1, -q0],
                  [-q2,  q3,  q0, -q1],
                  [q1, -q0,  q3, -q2]])
    quat_dot = 0.5 * G.T @ omega
    # Augment to maintain unit quaternion.
    quat_err = np.sum(quat**2) - 1
    quat_err_grad = 2 * quat
    quat_dot = quat_dot - quat_err * quat_err_grad
    return quat_dot


class Quadrotor():
    """
    Quadrotor forward dynamics model.
    """

    def __init__(self):
        # self.mass = 0.030  # kg nominal
        self.mass = 0.04  # kg true

        self.Ixx = 1.43e-5  # kg*m^2
        self.Iyy = 1.43e-5  # kg*m^2
        self.Izz = 2.89e-5  # kg*m^2
        self.arm_length = 0.046  # meters
        self.rotor_speed_min = 0  # rad/s
        self.rotor_speed_max = 2500  # rad/s
        self.k_thrust = 2.3e-08  # N/(rad/s)**2
        self.k_drag = 7.8e-11   # Nm/(rad/s)**2

        # Additional constants.
        self.inertia = np.diag(
            np.array([self.Ixx, self.Iyy, self.Izz]))  # kg*m^2
        self.g = 9.81  # m/s^2

        # Precomputes
        k = self.k_drag/self.k_thrust
        L = self.arm_length
        self.to_TM = np.array([[1,  1,  1,  1],
                               [0,  L,  0, -L],
                               [-L,  0,  L,  0],
                               [k, -k,  k, -k]])
        self.inv_inertia = inv(self.inertia)
        self.weight = np.array([0, 0, -self.mass*self.g])
        self.t_step = 0.01

        # Initialize state
        self.state = _unpack_state(np.zeros(13))
        self.uncertainty=np.zeros(13)

    def reset(self, position=[0, 0, 0], yaw=0, pitch=0, roll=0):
        '''
        state is a 13 dimensional vector
            postion*3 velocity*3 attitude(quaternion)*4 angular velocity*3
        state = [x y z dx dy dz qw qx qy qz r p q]
        dot_state = [dx dy dz ddx ddy ddz dqw dqx dqy dqz dr dp dq]
        '''
        s = np.zeros(13)
        s[0] = position[0]
        s[1] = position[1]
        s[2] = position[2]
        r = Rotation.from_euler('zxy', [yaw, roll, pitch], degrees=True)
        quat = r.as_quat()
        s[6] = quat[0]
        s[7] = quat[1]
        s[8] = quat[2]
        s[9] = quat[3]
        # the unassigned values of s are zeros
        self.state = _unpack_state(s)
        return self.state

    def step(self, cmd_rotor_speeds):
        '''
        Considering the max and min of rotor speeds
        action is a 4 dimensional vector: conmmand rotor speeds
        action = [w1, w2, w3, w4]
        '''
        rotor_speeds = np.clip(
            cmd_rotor_speeds, self.rotor_speed_min, self.rotor_speed_max)
        rotor_thrusts = self.k_thrust * rotor_speeds**2

        '''
        Next, [w1, w2, w3, w4] into [F Mx My Mz]
        '''
        TM = self.to_TM @ rotor_thrusts
        T = TM[0]  # u1
        M = TM[1:]  # u2

        # Form autonomous ODE for constant inputs and integrate one time step.

        def s_dot_fn(t, s):
            return self._s_dot_fn(t, s, T, M)
        '''
        The next state can be obtained through integration （Runge-Kutta）
        '''
        s = _pack_state(self.state)
        state_=self.state
        sol = scipy.integrate.solve_ivp(
            s_dot_fn, (0, self.t_step), s, first_step=self.t_step)
        s = sol['y'][:, -1]
        # turn state back to dict
        self.state = _unpack_state(s)
        ades=(self.state['v']-state_['v'])/self.t_step

        # Re-normalize unit quaternion.
        reward = 0
        done = 0
        info = {}
        return self.state, reward, done, info,ades

    def _s_dot_fn(self, t, s, u1, u2):
        """
        Compute derivative of state for quadrotor given fixed control inputs as
        an autonomous ODE.
        """

        state = _unpack_state(s)
        # page 73
        # Position derivative.
        x_dot = state['v']

        # Velocity derivative.
        F = u1 * Quadrotor.rotate_k(state['q'])
        v_dot = (self.weight + F) / self.mass

        # Orientation derivative.
        q_dot = quat_dot(state['q'], state['w'])

        # Angular velocity derivative. page 26 Equation 4
        omega = state['w']
        omega_hat = Quadrotor.hat_map(omega)
        w_dot = self.inv_inertia @ (u2 - omega_hat @ (self.inertia @ omega))

        # Pack into vector of derivatives.
        s_dot = np.zeros((13,))
        s_dot[0:3] = x_dot
        s_dot[3:6] = v_dot
        s_dot[6:10] = q_dot
        s_dot[10:13] = w_dot

        return s_dot
    
    def step_real(self, cmd_rotor_speeds):
        '''
        Considering the max and min of rotor speeds
        action is a 4 dimensional vector: conmmand rotor speeds
        action = [w1, w2, w3, w4]
        '''
        rotor_speeds = np.clip(
            cmd_rotor_speeds, self.rotor_speed_min, self.rotor_speed_max)
        rotor_thrusts = self.k_thrust * rotor_speeds**2

        '''
        Next, [w1, w2, w3, w4] into [F Mx My Mz]
        '''
       
        TM = self.to_TM @ rotor_thrusts
        T = TM[0]  # u1推力 F
        M = TM[1:]  # u2力矩 Mx My Mz

        # Form autonomous ODE for constant inputs and integrate one time step.

        def s_dot_fn_real(t, s):
            return self._s_dot_fn_real(t, s, T, M)
        '''
        The next state can be obtained through integration （Runge-Kutta）
        '''
        s = _pack_state(self.state)
        state_=self.state
        state_s=s
        # print("s",s)
        sol = scipy.integrate.solve_ivp(
            # s_dot_fn_real, (0, self.t_step), s+0.001*s, first_step=self.t_step)####x_t1=x_t+0.001x_t+dt*x_tdot
            s_dot_fn_real, (0, self.t_step), s, first_step=self.t_step)####x_t1=x_t+dt*x_tdot
        ####上面这里加上 nn
        s = sol['y'][:, -1]
        # self.uncertainty=0.001*state_s##
        # s=s+self.uncertainty

        # print("s",s)
        # ds_real=(s- _pack_state(self.state))/self.t_step
        # print("real_ds",ds_real)
        
        # turn state back to dict
        self.state = _unpack_state(s)
        ades=(self.state['v']-state_['v'])/self.t_step

        # Re-normalize unit quaternion.
        reward = 0
        done = 0
        info = {}
        return self.state, reward, done, info,ades

    def _s_dot_fn_real(self, t, s, u1, u2):
        """
        Compute derivative of state for quadrotor given fixed control inputs as
        an autonomous ODE.
        """

        state = _unpack_state(s)
        # page 73
        # Position derivative.
        x_dot = state['v']
        

        # Velocity derivative.
        # dis=np.array([0.5])
        # dis=np.array([0.04814792,0.05324697,0.4948197])
        F = (u1)* Quadrotor.rotate_k(state['q'])
        # -dbarx
        # print("dis",dis* Quadrotor.rotate_k(state['q']))
        v_dot = (self.weight + F) / self.mass

        # Orientation derivative.
        q_dot = quat_dot(state['q'], state['w'])
        # Angular velocity derivative. page 26 Equation 4
        omega = state['w']
        omega_hat = Quadrotor.hat_map(omega)
        w_dot = self.inv_inertia @ (u2 - omega_hat @ (self.inertia @ omega))

        # Pack into vector of derivatives.
        s_dot = np.zeros((13,))
        s_dot[0:3] = x_dot
        s_dot[3:6] = v_dot
        s_dot[6:10] = q_dot
        s_dot[10:13] = w_dot
       
        # print("s_dot",s_dot)
        return s_dot
    def get_dLTI(self, dt):
        num_x = 12
        Ac = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, self.g, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, (-self.g), 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], ])
        Bc = np.array([[0, 0, 0, 0],
                       [0, 0, 0, 0],
                       [0, 0, 0, 0],
                       [0, 0, 0, 0],
                       [0, 0, 0, 0],
                       [1/self.mass, 0, 0, 0],
                       [0, 0, 0, 0],
                       [0, 0, 0, 0],
                       [0, 0, 0, 0],
                       [0, 1/self.Ixx, 0, 0],
                       [0, 0, 1/self.Iyy, 0],
                       [0, 0, 0, 1/self.Izz]])
        Cc = np.eye(num_x)
        Dc = np.zeros((num_x, 4))
        sysc = control.ss(Ac, Bc, Cc, Dc)
        # Discretization
        sysd = control.sample_system(sysc, dt, method='bilinear')
        return sysd.A, sysd.B

    @classmethod
    def rotate_k(cls, q):
        """
        Rotate the unit vector k by quaternion q. This is the third column of
        the rotation matrix associated with a rotation by q.
        """
        return np.array([2 * (q[0] * q[2] + q[1] * q[3]),
                         2 * (q[1] * q[2] - q[0] * q[3]),
                         1 - 2 * (q[0] ** 2 + q[1] ** 2)])

    @classmethod
    def hat_map(cls, s):
        """
        Given vector s in R^3, return associate skew symmetric matrix S in R^3x3
        """
        return np.array([[0, -s[2], s[1]],
                         [s[2], 0, -s[0]],
                         [-s[1], s[0], 0]])


def _pack_state(state):
    """
    Convert a state dict to Quadrotor's private internal vector representation.
    """
    s = np.zeros((13,))
    s[0:3] = state['x'].squeeze()
    s[3:6] = state['v'].squeeze()
    s[6:10] = state['q'].squeeze()
    s[10:13] = state['w'].squeeze()
    return s


def _unpack_state(s):
    """
    Convert Quadrotor's private internal vector representation to a state dict.
    """
    state = {'x': s[0:3], 'v': s[3:6], 'q': s[6:10], 'w': s[10:13]}
    return state

