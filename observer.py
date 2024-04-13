import numpy as np
from numpy.linalg import inv, norm
import math
import quadrotor
import torch
from mpc_control_dob import instantiate_controller

class Observer:
    def __init__(self):
    # The constant parameters of quadrotor
        self.quad_model = quadrotor.Quadrotor()
        self.quad_controller=instantiate_controller()
        self.mass = 0.030  # kg
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
        k = self.k_drag/self.k_thrust#空气阻力系数与推力系数之比
        L = self.arm_length
        self.to_TM = np.array([[1,  1,  1,  1],
                               [0,  L,  0, -L],
                               [-L,  0,  L,  0],
                               [k, -k,  k, -k]])#将四个的马达推力转换为总的推力和扭矩
        self.inv_inertia = inv(self.inertia)#反转惯量矩阵
        self.weight = np.array([[0, 0, -self.mass*self.g]])
        self.t_step = 0.01

        # Initialize state
        self.state = np.zeros(13)

        self.Ks=0.2
        
        # self.Ks=2

        self.I=np.array([[1,0,0],
                         [0,1,0],
                         [0,0,1]])
        self.e3=np.array([0, 0, 1])
        # self.e3=self.e3.T
        self.zx=np.array([[0, 0, 0,0,0,0]])
        self.vdot=np.array([0, 0, 0])
        self.pos=np.array([0, 0, 0])
        self.xi=np.array([[0,0,0]])
        self.obs=[]
        self.distESO=[]
        self.distESOro=[]
        self.distDOB=[]
        self.dis=[]
        # self.l_gv=np.array([[0.  , 0.  , 0.  , 0.1 , 0.  , 0.  ],
        #                     [0.  , 0.  , 0.  , 0.1 , 0.  , 0.  ],
        #                     [0.  , 0.  , 0.  , 0.  , 0.08, 0.  ],
        #                     [0.  , 0.  , 0.  , 0.  , 0.08, 0.  ],
        #                     [0.  , 0.  , 0.  , 0.  , 0.  , 0.1 ],
        #                     [0.  , 0.  , 0.  , 0.  , 0.  , 0.1 ]])
        self.l_gv=np.array([[0.  , 0.  , 0.  , 0.2 , 0.  , 0.  ],
                            [0.  , 0.  , 0.  , 0.2 , 0.  , 0.  ],
                            [0.  , 0.  , 0.  , 0.  , 0.1, 0.  ],
                            [0.  , 0.  , 0.  , 0.  , 0.1, 0.  ],
                            [0.  , 0.  , 0.  , 0.  , 0.  , 0.2 ],
                            [0.  , 0.  , 0.  , 0.  , 0.  , 0.2 ]])
        self.px=np.array([[0,0,0,0,0,0]])
        self.A=np.array([[0,1/(2*math.pi)*np.sqrt(self.g/self.arm_length),0,0,0,0],
                         [-1/(2*math.pi)*np.sqrt(self.g/self.arm_length),0,0,0,0,0],
                         [0,0,0,1/(2*math.pi)*np.sqrt(self.g/self.arm_length),0,0],
                         [0,0,-1/(2*math.pi)*np.sqrt(self.g/self.arm_length),0,0,0],
                         [0,0,0,0,0,1/(2*math.pi)*np.sqrt(self.g/self.arm_length)],
                         [0,0,0,0,-1/(2*math.pi)*np.sqrt(self.g/self.arm_length),0]])
        self.C=np.array([[1, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0]])
        self.g2=np.array([[0,0,0],
                          [0,0,0],
                          [0,0,0],
                          [1/self.mass,0,0],
                          [0,1/self.mass,0],
                          [0,0,1/self.mass]])
        self.f=np.array([[0,0,0,0,0,-self.g]])

        self.k=np.array([[50,0,0,833,0,0,78,0,0],
                         [0,50,0,0,833,0,0,78,0],
                         [0,0,50,0,0,833,0,0,78]])
        
        self.zp1=np.array([[0,0,0]])
        self.zp2=np.array([[0,0,0]])
        self.zp3=np.array([[0,0,0]])

        self.za1=np.array([[0,0,0,0]])
        self.za2=np.array([[0,0,0]])
        self.za3=np.array([[0,0,0]])

        # self.ka=np.array([[50,0,0,833,0,0,3906,0,0],
        #                  [0,50,0,0,833,0,0,3906,0],
        #                  [0,0,50,0,0,833,0,0,3906]])
        self.ka=np.array([[50,0,0,833,0,0,0.4,0,0],
                         [0,50,0,0,833,0,0,0.4,0],
                         [0,0,50,0,0,833,0,0,0.4]])
        self.ka1=np.array([[50,0,0,0],
                          [0,50,0,0],
                          [0,0,50,0],
                          [0,0,0,50]])
        # self.ka2=np.array([[833,0,0,0],
        #                    [0,833,0,0],
        #                    [0,0,833,0]])
        # self.ka3=np.array([[0.4,0,0,0.1],
        #                    [0,0.4,0,0],
        #                    [0,0,0.4,0]])
        self.ka2=np.array([[833,0,0,0],
                 [0,833,0,0],
                 [0,0,833,0]])
        self.ka3=np.array([[0.4,0,0,0],
                   [0,0.4,0,0],
                   [0,0,0.4,0]])
        
    def observerESO(self, u1,dt, stateQ, xp,state13,u4,u_obs_DOB):#[ESO] thrust,t,q,x!!!!!!
        # 使用神经网络模型
        # ode_torch = torch.load("masschange_my_model_1000_13.pth", map_location=torch.device('cpu'))
        # ode_torch = torch.load("masschange0.05circle_my_model_1000_13_dia.pth", map_location=torch.device('cpu'))
        ode_torch = torch.load("R4masschange0.04circle_my_model_1000_13_dia.pth", map_location=torch.device('cpu'))

        param_ls = []
        for idx, layer in ode_torch.items():
            param_ls.append(layer.detach().cpu().numpy())

        activation = np.tanh
        ode_nn = np.append(state13, u4)

        n_layers = int(len(ode_torch)/2)
        param_cnt = 0 

        for i in range(n_layers):
            ode_nn = param_ls[param_cnt]@ode_nn + param_ls[param_cnt + 1]##线性层wx+b
            param_cnt += 2
            if (i + 1) != n_layers:
                ode_nn = activation(ode_nn)
        # f_ode_nn=np.zeros(6)
        # f_ode_nn[3:6]=ode_nn[3:6]
        f_ode_nn_v=np.expand_dims(ode_nn[0:3],axis=0)
        f_ode_nn_a=np.expand_dims(ode_nn[3:6],axis=0)
        
        #################   ESO   #################

        ep1=xp.T-self.zp1

        zp1dot=self.zp2+(self.k[:,:3]@ep1.T).T#+f_ode_nn_v
        

        F=u1 * self.quad_model.rotate_k(stateQ)
        F=np.expand_dims(F,axis=0)
        
        # du_obs_DOB=np.array([[0,0,u_obs_DOB[2]]])
        u_obs_DOB=np.expand_dims(u_obs_DOB,axis=0)
        
        # zp2dot=1/self.mass*(F+self.zp3+self.weight)+(self.k[:,3:6]@ep1.T).T
        # zp2dot=1/self.mass*(F+self.zp3+self.weight+u_obs_DOB)+(self.k[:,3:6]@ep1.T).T+f_ode_nn_a  ##+u_obs_DOB+f_ode_nn_a
        zp2dot=1/self.mass*(F+self.zp3+self.weight+u_obs_DOB)+(self.k[:,3:6]@ep1.T).T  ##+u_obs_DOB
        # zp2dot=1/self.mass*(F+self.zp3+self.weight)+(self.k[:,3:6]@ep1.T).T+f_ode_nn_a

        zp3dot=(self.k[:,6:]@ep1.T).T
        
        #扰动估计
        
        # self.dbarxz=self.dbarx[2]
        self.zp1=self.zp1+zp1dot*dt
        # print("xp1",xp.T)
        # print("self.zp1",self.zp1)
        self.zp2=self.zp2+zp2dot*dt
        # print("self.zp2",self.zp2)
        self.zp3=self.zp3+zp3dot*dt
        self.dbarx = self.zp3
        # print("dbarx",self.dbarx)
        self.distESO.append(self.dbarx.squeeze())
        # print("eso",self.distESO)

        return self.distESO
    
    def observerDOB(self, u1,dt, state, vdes,ades,state13,u4,u_obs_ESO):#[DOB]thrust,t,q,v,a!!!!!!
        # 使用神经网络模型
        # ode_torch = torch.load("masschange_my_model_1000_13.pth", map_location=torch.device('cpu'))
        # ode_torch = torch.load("masschange0.05circle_my_model_1000_13_dia.pth", map_location=torch.device('cpu'))
        ode_torch = torch.load("R4masschange0.04circle_my_model_1000_13_dia.pth", map_location=torch.device('cpu'))

        param_ls = []
        for idx, layer in ode_torch.items():
            param_ls.append(layer.detach().cpu().numpy())

        activation = np.tanh
        ode_nn = np.append(state13, u4)

        n_layers = int(len(ode_torch)/2)
        param_cnt = 0 

        for i in range(n_layers):
            ode_nn = param_ls[param_cnt]@ode_nn + param_ls[param_cnt + 1]##线性层wx+b
            param_cnt += 2
            if (i + 1) != n_layers:
                ode_nn = activation(ode_nn)
        f_ode_nn=np.zeros(6)
        f_ode_nn[3:6]=ode_nn[3:6]##取vdot，xdot已在v中被补偿
        f_ode_nn=np.expand_dims(f_ode_nn,axis=0)
        #################   DOB   #################

        vdes=np.expand_dims(vdes,axis=0)
        ades=np.expand_dims(ades,axis=0)
        gv=np.concatenate((vdes,ades),axis=1)
        self.px = self.px+(self.l_gv@gv.T*dt).T
        
        F=u1 * self.quad_model.rotate_k(state)
        F=np.expand_dims(F,axis=0)
        self.f[0,:3]=vdes


        f_normal=self.f.T+ self.g2@F.T
        f_model=f_normal+f_ode_nn.T

        u_obs_ESO=np.expand_dims(u_obs_ESO,axis=0)
        # self.zxdot=(self.A-self.l_gv@self.g2@self.C)@(self.zx.T)+self.A@self.px.T-self.l_gv@(self.g2@self.C@self.px.T + f_model+self.g2@u_obs_ESO.T)##
        self.zxdot=(self.A-self.l_gv@self.g2@self.C)@(self.zx.T)+self.A@self.px.T-self.l_gv@(self.g2@self.C@self.px.T + f_normal+self.g2@u_obs_ESO.T)##
        # self.zxdot=(self.A-self.l_gv@self.g2@self.C)@(self.zx.T)+self.A@self.px.T-self.l_gv@(self.g2@self.C@self.px.T + f_model)##
        # self.zxdot=(self.A-self.l_gv@self.g2@self.C)@(self.zx.T)+self.A@self.px.T-self.l_gv@(self.g2@self.C@self.px.T + self.f.T+ self.g2@F.T)
        self.zxdot=self.zxdot.T
        #状态估计
        self.zx = self.zx + self.zxdot*dt
        # print("x_ob",self.zx)
        

        #扰动估计
        self.xi = (self.zx.T + self.px.T).T
        self.dbarx = (self.C@self.xi.T).T
        # print("dbarx",self.dbarx)

        # self.obs.append(self.zx.squeeze())
        self.distDOB.append(self.dbarx.squeeze())
        # print("dob",self.distDOB)

        return self.distDOB

    def observerESOro(self, u2,dt, stateq, xp, state13, u4):#[ESOro] u2,t,q四元数,phi,theta,psi!!!!!!
        # 使用神经网络模型
        # ode_torch = torch.load("masschange_my_model_1000_13.pth", map_location=torch.device('cpu'))
        # ode_torch = torch.load("masschange0.05circle_my_model_1000_13_dia.pth", map_location=torch.device('cpu'))
        ode_torch = torch.load("R4masschange0.04circle_my_model_1000_13_dia.pth", map_location=torch.device('cpu'))

        param_ls = []
        for idx, layer in ode_torch.items():
            param_ls.append(layer.detach().cpu().numpy())

        activation = np.tanh
        ode_nn = np.append(state13, u4)

        n_layers = int(len(ode_torch)/2)
        param_cnt = 0 

        for i in range(n_layers):
            ode_nn = param_ls[param_cnt]@ode_nn + param_ls[param_cnt + 1]##线性层wx+b
            param_cnt += 2
            if (i + 1) != n_layers:
                ode_nn = activation(ode_nn)
        # f_ode_nn=np.zeros(6)
        # f_ode_nn[3:6]=ode_nn[3:6]
        f_ode_nn_dq=np.expand_dims(ode_nn[6:10],axis=0)
        f_ode_nn_dw=np.expand_dims(ode_nn[10:],axis=0)

        #################   ESOro   #################za1=q0q1q2q3,za2=pqr,za3=disturbance
        phi=self.za1[0,0]
        theta=self.za1[0,1]
        psi=self.za1[0,2]
        
        # W_yita=np.array([[1,math.sin(phi)*math.tan(theta),math.cos(phi)*math.tan(theta)],
        #                   [0,math.cos(phi),-math.sin(phi)],
        #                   [0,math.sin(phi)/math.cos(theta),math.cos(phi)/math.cos(theta)]])
       
        ep1=stateq.T-self.za1 

        G = np.array([[stateq[3],  stateq[2], -stateq[1], -stateq[0]],
                  [-stateq[2],  stateq[3],  stateq[0], -stateq[1]],
                  [stateq[1], -stateq[0],  stateq[3], -stateq[2]]])
        # G = np.array([[-stateq[1],  stateq[0], -stateq[3], stateq[2]],
        #           [-stateq[2],  stateq[3],  stateq[0], -stateq[1]],
        #           [-stateq[3], -stateq[2],  stateq[1], stateq[0]]])
        
        # za1dot=(0.5*(G.T @ self.za2.T).T-2*(np.sum(stateq**2) - 1)* stateq)+(self.ka1@ep1.T).T#+f_ode_nn_dq
        za1dot=(0.5*(G.T @ self.za2.T).T)+(self.ka1@ep1.T).T#+f_ode_nn_dq
        omega_hat = self.quad_model.hat_map(self.za2.squeeze())
        wjw=omega_hat @ (self.inertia @ self.za2.T)
        JW=self.inv_inertia @ (u2 - wjw.T+self.za3).T
        za2dot=JW+(self.ka2@ep1.T)
        za2dot=za2dot.T#+f_ode_nn_dw
        za3dot=(self.ka3@ep1.T).T
        
        #扰动估计
        self.za1=self.za1+za1dot*dt
        # print("self.zp1",self.zp1)
        self.za2=self.za2+za2dot*dt
        # print("self.zp2",self.zp2)
        self.za3=self.za3+za3dot*dt
        self.dbarx = self.za3
        # print("dbarx",self.dbarx)
        self.distESOro.append(self.dbarx.squeeze())
        # print("eso",self.distESO)

        #################   ESOro   #################zp1=phi;theta;psi,zp2=pqr,zp3=disturbance
        # phi=self.za1[0,0]
        # theta=self.za1[0,1]
        # psi=self.za1[0,2]
        
        # W_yita=np.array([[1,math.sin(phi)*math.tan(theta),math.cos(phi)*math.tan(theta)],
        #                   [0,math.cos(phi),-math.sin(phi)],
        #                   [0,math.sin(phi)/math.cos(theta),math.cos(phi)/math.cos(theta)]])
       
        # ep1=xp.T-self.za1 

        # za1dot=(W_yita @ self.za2.T).T+(self.ka[:,:3]@ep1.T).T#+f_ode_nn_dphithetapsi
        # omega_hat = self.quad_model.hat_map(self.za2.squeeze())
        # wjw=omega_hat @ (self.inertia @ self.za2.T)
        # JW=self.inv_inertia @ (u2 - wjw.T+self.za3).T
        # za2dot=JW+(self.ka[:,3:6]@ep1.T)
        # za2dot=za2dot.T+f_ode_nn_dw
        # za3dot=(self.ka[:,6:]@ep1.T).T
        
        # #扰动估计
        # self.za1=self.za1+za1dot*dt#+f_ode_nn_phithetapsi
        # # print("self.zp1",self.zp1)
        # self.za2=self.za2+za2dot*dt
        # # print("self.zp2",self.zp2)
        # self.za3=self.za3+za3dot*dt
        # self.dbarx = self.za3
        # print("dbarx",self.dbarx)
        # self.distESOro.append(self.dbarx.squeeze())
        # # print("eso",self.distESO)

        #################   ESOro   #################zp1=phi;theta;psi,zp2=\dot,zp3=disturbance
        # ep1=xp-self.zp1 

        # zp1dot=self.zp2+(self.ka[:,:3]@ep1.T).T#+f_ode_nn_v

        # phi=self.zp1[0,0]
        # theta=self.zp1[0,1]
        # psi=self.zp1[0,2]
        # M=np.array([[self.Ixx,0,-self.Ixx*math.sin(theta)],
        #         [0,self.Iyy*math.cos(phi)*math.cos(phi)+self.Izz*math.sin(phi)*math.sin(phi),(self.Iyy-self.Izz)*math.cos(phi)*math.sin(phi)*math.cos(theta)],
        #         [-self.Ixx*math.sin(theta),(self.Iyy-self.Izz)*math.cos(phi)*math.sin(phi)*math.cos(theta),self.Ixx*math.sin(theta)*math.sin(theta)+(self.Iyy*math.sin(phi)*math.sin(phi)+self.Izz*math.cos(phi)*math.cos(phi))*math.cos(theta)*math.cos(theta)]])
        # dphi=self.zp2[0,0]
        # dtheta=self.zp2[0,1]
        # dpsi=self.zp2[0,2]
        # C=np.array([[0,(self.Iyy-self.Izz)*(dtheta*math.cos(phi)*math.sin(phi)+dpsi*math.sin(phi)*math.sin(phi)*math.cos(theta))-(self.Ixx+self.Iyy*math.cos(phi)*math.cos(phi)-self.Izz*math.cos(phi)*math.cos(phi))*dpsi*math.cos(theta),(self.Izz-self.Iyy)*dpsi*math.cos(phi)*math.sin(phi)*math.cos(theta)*math.cos(theta)],
        #         [(self.Izz-self.Iyy)*(dtheta*math.cos(phi)*math.sin(phi)+dpsi*math.sin(phi)*math.sin(phi)*math.cos(theta))+(self.Ixx+self.Iyy*math.cos(phi)*math.cos(phi)-self.Izz*math.cos(phi)*math.cos(phi))*dpsi*math.cos(theta),(self.Izz-self.Iyy)*dphi*math.cos(phi)*math.sin(phi),(-self.Ixx+self.Iyy*math.sin(phi)*math.sin(phi)+self.Izz*math.cos(phi)*math.cos(phi))*dpsi*math.sin(theta)*math.cos(theta)],
        #         [(self.Iyy-self.Izz)*dpsi*math.cos(phi)*math.sin(phi)*math.cos(theta)*math.cos(theta)-self.Ixx*dtheta*math.cos(theta),(self.Izz-self.Iyy)*(dtheta*math.cos(phi)*math.sin(phi)*math.sin(theta)+dphi*math.sin(phi)*math.sin(phi)*math.cos(theta)-dphi*math.cos(phi)*math.cos(phi)*math.cos(theta))+(self.Ixx-self.Iyy*math.sin(phi)*math.sin(phi)-self.Izz*math.cos(phi)*math.cos(phi))*dpsi*math.sin(theta)*math.cos(theta),(self.Iyy-self.Izz)*dphi*math.cos(phi)*math.sin(phi)*math.cos(theta)*math.cos(theta)+(self.Ixx-self.Iyy*math.sin(phi)*math.sin(phi)-self.Izz*math.cos(phi)*math.cos(phi))*dtheta*math.cos(theta)*math.sin(theta)]])
        # W_yita=np.array([[1,math.sin(phi)*math.tan(theta),math.cos(phi)*math.tan(theta)],
        #                   [0,math.cos(phi),-math.sin(phi)],
        #                   [0,math.sin(phi)/math.cos(theta),math.cos(phi)/math.cos(theta)]])
        # statew=np.expand_dims(statew,axis=0)
        # zp2dot=(W_yita@statew.T).T+(self.ka[:,3:6]@ep1.T).T
        # # ro=(u2-(self.zp2@C.T)+self.zp3)@(np.linalg.pinv(M)).T

        # # zp2dot=ro+(self.ka[:,3:6]@ep1.T).T
        # zp3dot=(self.ka[:,6:]@ep1.T).T
        
        # #扰动估计
        # self.zp1=self.zp1+zp1dot*dt
        # # print("xp1",xp.T)
        # # print("self.zp1",self.zp1)
        # self.zp2=self.zp2+zp2dot*dt
        # # print("self.zp2",self.zp2)
        # self.zp3=self.zp3+zp3dot*dt
        # self.dbarx = self.zp3
        
        # print("dbarx",self.dbarx)
        # self.distESO.append(self.dbarx.squeeze())
        # # print("eso",self.distESO)

        return self.distESOro 
    #########################################################
    # def observer(self, u1,dt, state, vdes):#u1=u_mpc+u_dis!!!!!!
    #     #############constant disturbance#################
    #     vdes=vdes.T[0]
    #     lv = self.Ks * self.I
    #     px = self.Ks * vdes
        
    #     F=u1 * self.quad_model.rotate_k(state)
        
    #     self.zxdot=-lv @ (1/self.mass * (px + self.zx) - (self.weight + F)/self.mass).T
    

    #     #扰动估计
    #     self.dbarx = self.zx + px
    #     print("dbarx",self.dbarx)
    #     # self.dbarxz=self.dbarx[2]
        
        
    #     #状态估计
    #     self.zx = self.zx + self.zxdot*dt
        
    #     print("x_ob",self.zx)

    #     self.obs.append(self.zx)
    #     self.dist.append(self.dbarx)
        
    #     return self.obs,self.dist
    ########################################################
        
    # def observer(self, u1,dt, state):#无观测器
        
    #     F=u1 * self.quad_model.rotate_k(state)
    #     zxdot=(self.weight + F)/self.mass

    #     # ddx = (thrust)/self.mq*np.sin(the)    
    #     # ddy = -(thrust)/self.mq*np.sin(phi)
    #     # ddz = self.g - (thrust)/self.mq
    #     # zxdot = np.array([ddx, ddy, ddz])

        
    #     #状态估计
    #     self.zx = self.zx + zxdot*dt##速度而不是位置
    #     self.pos = self.pos + self.zx*dt
    #     print("x_ob",self.pos)
        