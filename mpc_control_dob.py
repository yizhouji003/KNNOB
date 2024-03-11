import torch
from quadrotor import Quadrotor
# from Simulation.crazyflie_params import quad_params
from casadi import *
import numpy as np
from delta_nn import control_learner
import quadrotor
import math

##MPC
class Control(object):
    def __init__(self):
        quad = Quadrotor()
        # self.mass = quad.mass
        self.mass = 0.03 #nominal mass

        self.g = 9.81  # m/s^2
        self.weight=self.mass*self.g

        self.length = quad.arm_length
        self.rotor_min = quad.rotor_speed_min
        self.rotor_max = quad.rotor_speed_max
        self.k_t = quad.k_thrust
        self.k_d = quad.k_drag

        self.inertia = np.diag(np.array([quad.Ixx, quad.Iyy, quad.Izz]))
        self.inv_inertia = np.linalg.inv(self.inertia)
        k = self.k_d / self.k_t
        self.cf_map = np.array([[1, 1, 1, 1],
                               [0, self.length, 0, -self.length],
                               [-self.length, 0, self.length, 0],
                               [k, -k, k, -k]])
        self.fc_map = np.linalg.inv(self.cf_map)
        self.motor_spd = 1790.0
        force = self.k_t * np.square(self.motor_spd)
        self.forces_old = np.array([force, force, force, force])

        self.init_mpc = 0
        self.x = np.zeros((1,))
        self.g = np.zeros((1,))
        self.quad_model = quadrotor.Quadrotor()
        

    def generate_OB_control_input(self, u_mpc,u_obs_ESO,stateq,u_dis_DO,u_obs_DO,obs_obs_ro):
        cmd_motor_speeds = np.zeros((4,))
        cmd_thrust = 0
        cmd_moment = np.zeros((3,))

        rotate=np.expand_dims(self.quad_model.rotate_k(stateq),axis=0)###
        obs_rotate=u_obs_ESO @ np.linalg.pinv(rotate)

        u_compensateESO=np.zeros(4)
        u_compensateESO[0]=obs_rotate
        u_diss_ESO=np.zeros(4)
        u_diss_ESO[0]=np.array([0.5])
        # u_diss_ESO[0]=np.array([0.9])

        u_compensate_DO=np.zeros(4)
        dob_rotate=u_obs_DO @ np.linalg.pinv(rotate)
        u_compensate_DO[0]=dob_rotate
        u_diss_DO=np.zeros(4)
        u_diss_DO[0]=u_dis_DO[0,0]
        
        u_diss_ro=np.zeros(4)
        u_diss_ro[1:]=np.array([1e-3,1e-4,1e-5])
        # u_diss_ro[1:]=np.array([1e-4,1e-4,1e-4])
        # u_diss_ro[1:]=np.array([1e-4,1e-5,1e-6])

        u_compensate_ro=np.zeros(4)
        u_compensate_ro[1:]=obs_obs_ro

        u=u_mpc+u_diss_DO-u_compensate_DO+u_diss_ro-u_compensate_ro+u_diss_ESO-u_compensateESO
        print("u_diss_ro-u_compensate_ro",u_diss_ro-u_compensate_ro)
        print("u_compensateESO",u_compensateESO)
        print("err",u_diss_DO-u_compensate_DO)
        forces = np.squeeze(self.fc_map @ np.squeeze(np.array([u])))#4
        
        for i in range(4): 
            if forces[i] < 0:
                    forces[i] = 0
                    cmd_motor_speeds[i] = self.rotor_min
            cmd_motor_speeds[i] = np.sqrt(forces[i] / self.k_t)
            if cmd_motor_speeds[i] > self.rotor_max:
                cmd_motor_speeds[i] = self.rotor_max
        cmd_desire=u_mpc-u_compensate_DO-u_compensate_ro-u_compensateESO
        
        cmd_thrust = u[0]  # thrust
        cmd_moment[0] = u[1]  # moment about p
        cmd_moment[1] = u[2]  # moment about q
        cmd_moment[2] = u[3]  # moment about r

        control_input = { 'cmd_desire':cmd_desire,
                         'cmd_motor_speeds': cmd_motor_speeds,
                         'cmd_thrust': cmd_thrust,
                         'cmd_moment': cmd_moment}
        return control_input
    
    # def generate_ESO_control_input(self, u_mpc,u_obs,stateq):
    #     cmd_motor_speeds = np.zeros((4,))
    #     cmd_thrust = 0
    #     cmd_moment = np.zeros((3,))

    #     rotate=np.expand_dims(self.quad_model.rotate_k(stateq),axis=0)###
    #     obs_rotate=u_obs @ np.linalg.pinv(rotate)

    #     u_compensate=np.zeros(4)
    #     u_compensate[0]=obs_rotate
    #     # print("com",u_compensate)

    #     u_diss=np.zeros(4)
    #     # u_diss[0]=np.array([0.5])
    #     u_diss[0]=np.array([0.8])

    #     u=u_mpc+u_diss-u_compensate###ESO
    #     print(u_diss-u_compensate)

    #     forces = np.squeeze(self.fc_map @ np.squeeze(np.array([u])))#4
        
    #     for i in range(4):
    #         if forces[i] < 0:
    #                 forces[i] = 0
    #                 cmd_motor_speeds[i] = self.rotor_min
    #         cmd_motor_speeds[i] = np.sqrt(forces[i] / self.k_t)
    #         if cmd_motor_speeds[i] > self.rotor_max:
    #             cmd_motor_speeds[i] = self.rotor_max
    #     cmd_desire=u_mpc-u_compensate
        
    #     cmd_thrust = u[0]  # thrust
    #     cmd_moment[0] = u[1]  # moment about p
    #     cmd_moment[1] = u[2]  # moment about q
    #     cmd_moment[2] = u[3]  # moment about r

    #     control_input = { 'cmd_desire':cmd_desire,
    #                      'cmd_motor_speeds': cmd_motor_speeds,
    #                      'cmd_thrust': cmd_thrust,
    #                      'cmd_moment': cmd_moment}
    #     return control_input
    
    # def generate_DOB_control_input(self, u_mpc,u_dis,u_obs):
    #     cmd_motor_speeds = np.zeros((4,))
    #     cmd_thrust = 0
    #     cmd_moment = np.zeros((3,))
        
    #     u_compensate=np.zeros(4)
    #     u_diss=np.zeros(4)
    #     u_compensate[0]=u_obs[2]###DOB
    #     u_diss[0]=u_dis[0,0]
    #     u=u_mpc+u_diss-u_compensate###DOB

    #     forces = np.squeeze(self.fc_map @ np.squeeze(np.array([u])))
    #     # forces[forces < 0] = np.square(self.forces_old[forces < 0]) * self.k_t
    #     # cmd_motor_speeds = np.clip(np.sqrt(forces / self.k_t), self.rotor_min, self.rotor_max)
    #     # output = self.cf_map @ (self.k_t * np.square(cmd_motor_speeds))
    #     # self.forces_old = forces
    #     for i in range(4):
    #         if forces[i] < 0:
    #                 forces[i] = 0
    #                 cmd_motor_speeds[i] = self.rotor_min
    #         cmd_motor_speeds[i] = np.sqrt(forces[i] / self.k_t)
    #         if cmd_motor_speeds[i] > self.rotor_max:
    #             cmd_motor_speeds[i] = self.rotor_max

    #     cmd_desire=u_mpc-u_compensate###DOB

    #     cmd_thrust = u[0]  # thrust
    #     cmd_moment[0] = u[1]  # moment about p
    #     cmd_moment[1] = u[2]  # moment about q
    #     cmd_moment[2] = u[3]  # moment about r

    #     control_input = { 'cmd_desire':cmd_desire,
    #                      'cmd_motor_speeds': cmd_motor_speeds,
    #                      'cmd_thrust': cmd_thrust,
    #                      'cmd_moment': cmd_moment
    #                      }
    #     return control_input
    
###更新控制输入u
    def update(self, state, flat,u_dis,obs_compensate_DOB,obs_compensate_ESO,obs_compensate_ro):
    # def update(self, state, flat,obs_compensate_ESO):
        opti = Opti()
        x = opti.variable(13, 21)
        u = opti.variable(4, 20)

        error_pos = (flat.get('x') - state.get('x').reshape(3, 1)).flatten()

        opti.minimize(1*sumsqr(x[0:3, :] - flat['x']) + 1*sumsqr(x[3:6, :] - flat['v']) +
                      1*sumsqr(x[6:9, :]) + 1*sumsqr(x[9, :] - 1.0) + 1*sumsqr(x[10:13, :]) + 1*sumsqr(u))
        for k in range(20):
            opti.subject_to(x[:, k + 1] == self.Dynamics(x[:, k], u[:, k]))

        opti.subject_to(opti.bounded(0.0, u[0, :], 0.575))##u_bound
        
        
        opti.subject_to(x[:, 0] == vertcat(state['x'], state['v'], state['q'], state['w']))

        opti.solver("ipopt", dict(print_time=False), dict(print_level=0, warm_start_init_point='yes'))

        if self.init_mpc >= 1:
            opti.set_initial(opti.x, self.x)
            opti.set_initial(opti.lam_g, self.g)

        sol = opti.solve()
        
        ###
        # control_input = self.generate_DOB_control_input(sol.value(u[:, 0]),u_dis,obs_compensate_DOB)##DOB
        # control_input = self.generate_ESO_control_input(sol.value(u[:, 0]),obs_compensate_ESO,state['q'])##ESO
        control_input = self.generate_OB_control_input(sol.value(u[:, 0]),obs_compensate_ESO,state['q'],u_dis,obs_compensate_DOB,obs_compensate_ro)##
        
        self.init_mpc += 1
        self.x = sol.value(opti.x)
        self.g = sol.value(opti.lam_g)
        # control_input = {'cmd_desire':output,
        #                  'cmd_thrust': output[0], 
        #                  'cmd_moment': output[1:]}
        return control_input,error_pos
####更新模型
    def update_model(self):
        x = MX.sym('x', 13, 1)
        u = MX.sym('u', 4, 1)

        pqr_vec = vertcat(x[10], x[11], x[12])
        G_transpose = horzcat(vertcat(x[9], x[8], -x[7], -x[6]), vertcat(-x[8], x[9], x[6], -x[7]),
                              vertcat(x[7], -x[6], x[9], -x[8]))
        quat_dot = 0.5 * mtimes(G_transpose, pqr_vec)##求四元数导数
        ode_without_u = vertcat(vertcat(x[3], x[4], x[5]), vertcat(0, 0, -9.81), quat_dot,
                                mtimes(self.inv_inertia, (-cross(pqr_vec, mtimes(self.inertia, pqr_vec)))))

        xdotdot_u = vertcat(2 * (x[6] * x[8] + x[7] * x[9]), 2 * (x[7] * x[8] - x[6] * x[9]),
                            (1 - 2 * (x[6] ** 2 + x[7] ** 2))) / self.mass * u[0]
        u_component = vertcat([0, 0, 0], xdotdot_u, [0, 0, 0, 0],
                              mtimes(self.inv_inertia, (vertcat(u[1], u[2], u[3]))))
        ##xdot,vdot,qdot,wdot
        nominal_model = ode_without_u + u_component

        # 使用神经网络模型
        ode_torch = torch.load("masschange_my_model_1000_13.pth", map_location=torch.device('cpu'))
        # ode_torch = torch.load("masschange0.05circle_my_model_1000_13_dia.pth", map_location=torch.device('cpu'))
        # ode_torch = torch.load("masschange0.04circle_my_model_1000_13_dia.pth", map_location=torch.device('cpu'))


        param_ls = []
        for idx, layer in ode_torch.items():
            param_ls.append(layer.detach().cpu().numpy())

        activation = tanh
        ode_nn = vertcat(x, u)

        n_layers = int(len(ode_torch)/2)
        param_cnt = 0
        hybrid_model =  nominal_model    

        for i in range(n_layers):
            ode_nn = mtimes(param_ls[param_cnt], ode_nn) + param_ls[param_cnt + 1]##线性层wx+b
            param_cnt += 2
            if (i + 1) != n_layers:
                ode_nn = activation(ode_nn)

        hybrid_model[:13] = hybrid_model[:13] +  ode_nn


        f = Function('f', [x, u], [hybrid_model])##混合模型的新函数 f
        intg = integrator('intg', 'rk', {'x': x, 'p': u, 'ode': f(x, u)},
                          dict(tf=0.05, simplify=True, number_of_finite_elements=4))
        
        res = intg(x0=x, p=u)
        self.Dynamics = Function('F', [x, u], [res['xf']])##混合模型的输出

        
        return 1
    
    def euler_from_quaternion(self, x, y, z, w):###q*4转换为phi,theta,psi
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)

        return np.array([roll_x, pitch_y, yaw_z])  # in radians


def instantiate_controller():
    controller = Control()
    controller.update_model()
    return controller
