import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import random_split
import pytorch_lightning as pl
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.autograd.functional as AGF
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning import loggers as pl_loggers
from numpy import cos, sin, arccos, arctan2, sqrt
from torch.utils.data import TensorDataset
import numpy as np

from torch.optim.lr_scheduler import StepLR
from quadrotor import Quadrotor
from numpy.linalg import inv, norm
import scipy.integrate
from scipy.spatial.transform import Rotation

class delta_nn(nn.Module):
    def __init__(self):
        super(delta_nn,self).__init__()
        self.w01=nn.Linear(17,64)
        # self.w02=nn.Linear(64,64)
        # self.w03=nn.Linear(64,64)
        self.w04=nn.Linear(64,32)
        self.w05=nn.Linear(32,13)
        
        # nn.init.xavier_uniform_(self.w01.weight,0.01)
        # nn.init.xavier_uniform_(self.w02.weight,0.01)
        # nn.init.xavier_uniform_(self.w03.weight,0.01)
        # nn.init.constant_(self.w01.bias,0)
        # nn.init.constant_(self.w02.bias,0)
        # nn.init.constant_(self.w03.bias,0)


    def forward(self,x,u):
        x1=x.clone()
        u1=u.clone()
        z=torch.cat((x1,u1),dim=0)
        z1=z.float()
        z01 = self.w01(z1)
        z011 = torch.tanh(z01)
        # z012 = self.w02(z011)
        # z013 = torch.tanh(z012)
        # z014 = self.w03(z013)
        # z015 = torch.tanh(z014)
        z012 = self.w04(z011)
        z013 = torch.tanh(z012)
        z_out = self.w05(z013)

        

        
        return z_out
    
    
class control_learner(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.controller=delta_nn()
        
        
    def forward(self, x):
        return 0

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam([{'params':self.controller.parameters()}]
        #     , lr=1e-3)
        optimizer = torch.optim.Adam([{'params':self.controller.parameters()}]
            , lr=8e-4)
        # scheduler=StepLR(optimizer,step_size=100,gamma=0.99)
        return [optimizer]#,[scheduler]

    # def training_step(self, train_batch, batch_idx):
    #     x,u,x_reality,f_xu=train_batch[:,:13],train_batch[:,13:17],train_batch[:,17:30],train_batch[:,30:]
    
        
    #     loss = torch.zeros(1).to(self.device)
    #     # print(x.shape)
    #     # print(u.shape)
    #     # train_batch represents initial states
    #     num_traj = train_batch.shape[0]
    #     y=torch.zeros((num_traj,13)).to(self.device)
    #     error=torch.zeros((num_traj,13)).to(self.device)
    #     # print(num_traj)
    #     for i in range(num_traj):
    #         # error=torch.zeros(1).to(self.device)
    #         y[i]=f_xu[i,:]+self.controller(x[i,:],u[i,:])#13
    #         error=x_reality[i,:]-y[i]
    #         loss = loss + error.t()@error
    #     # loss=loss/num_traj
    #     self.log('train_loss', loss)
    #     # print("loss",loss)
    #     return loss
    
    def xt1_ideal(self, x, u):
        self.Ixx = 1.43e-5  # kg*m^2
        self.Iyy = 1.43e-5  # kg*m^2
        self.Izz = 2.89e-5  # kg*m^2
        self.inertia = torch.diag(
            torch.tensor([self.Ixx, self.Iyy, self.Izz]))  # kg*m^2
        self.inv_inertia = torch.linalg.inv(self.inertia)
        self.t_step=0.01
        self.mass = 0.030  # kg
        self.g = 9.81  # m/s^2
        self.weight = torch.tensor([0, 0, -self.mass*self.g])

        def rotate_k(q):
            return torch.tensor([2 * (q[0] * q[2] + q[1] * q[3]),
                         2 * (q[1] * q[2] - q[0] * q[3]),
                         1 - 2 * (q[0] ** 2 + q[1] ** 2)])
        
        def hat_map(s):
            return torch.tensor([[0, -s[2], s[1]],
                            [s[2], 0, -s[0]],
                            [-s[1], s[0], 0]])
        
        u1=u[:1].float()
        u2=u[1:].float()

        def s_dot_fn(t, s):
            return _s_dot_fn(self,t, s, u1, u2)
        
        def quat_dot(quat, omega):
            # Adapted from "Quaternions And Dynamics" by Basile Graf.
            (q0, q1, q2, q3) = (quat[0], quat[1], quat[2], quat[3])
            G = torch.tensor([[q3,  q2, -q1, -q0],
                        [-q2,  q3,  q0, -q1],
                        [q1, -q0,  q3, -q2]])
            quat_dot = 0.5 * G.T @ omega
            # Augment to maintain unit quaternion.
            quat_err = torch.sum(quat**2) - 1
            quat_err_grad = 2 * quat
            quat_dot = quat_dot - quat_err * quat_err_grad
            return quat_dot

        def _s_dot_fn(self, s, u1, u2):
            
            # state = {'x': s[0:3], 'v': s[3:6], 'q': s[6:10], 'w': s[10:13]}
            s=torch.tensor(s).float()
            # Position derivative.
            x_dot = s[3:6]

            # Velocity derivative.
            F = u1 * rotate_k(s[6:10])
            v_dot = (self.weight + F) / self.mass

            # Orientation derivative.
            q_dot = quat_dot(s[6:10], s[10:13])

            # Angular velocity derivative. page 26 Equation 4
            omega = s[10:13]
            omega_hat = hat_map(omega)
            w_dot = self.inv_inertia @ (u2 - omega_hat @ (self.inertia @ omega))

            # Pack into vector of derivatives.
            s_dot = torch.zeros((13,))
            s_dot[0:3] = x_dot
            s_dot[3:6] = v_dot
            s_dot[6:10] = q_dot
            s_dot[10:13] = w_dot

            return s_dot

        s = x.float()##chuan 13
        
        # sol = scipy.integrate.solve_ivp(
        #     s_dot_fn, (0, self.t_step), s, first_step=self.t_step)
        # s = sol['y'][:, -1]

        s=s+_s_dot_fn(self,s, u1, u2)*self.t_step
        ##13转12
        # q = s[6:10]
        # w = s[10:13]
        # euler_ang = self.euler_from_quaternion(q[0], q[1], q[2], q[3])
        # euler_ang = np.zeros(3)
        # euler_ang[2] = state.get('yaw')
        # w = np.zeros(3)
        # w[2] = state.get('yaw_dot')

        # s = np.block([x, v, euler_ang, w])
        return s

    def xt1_ideal_12(self,x,u):
        self.Ixx = 1.43e-5  # kg*m^2
        self.Iyy = 1.43e-5  # kg*m^2
        self.Izz = 2.89e-5  # kg*m^2
        self.inertia = torch.diag(
            torch.tensor([self.Ixx, self.Iyy, self.Izz]))  # kg*m^2
        self.inv_inertia = torch.linalg.inv(self.inertia)
        self.t_step=0.01
        self.mass = 0.030  # kg
        self.g = 9.81  # m/s^2
        self.weight = torch.tensor([0, 0, -self.mass*self.g])
        u1=u[:1].float()
        u_roll=u[1].float()
        u_pitch=u[2].float()
        u_yaw=u[3].float()

        def _s_dot_fn(self, s, u1, u_roll,u_pitch,u_yaw):
            v_dot=torch.zeros(3)
            x_dot = s[3:6]
            v_dot[0]=(cos(s[8]) * sin(s[7]) + cos(s[7]) * sin(s[6]) * sin(s[8])) * u1
            v_dot[1]=(sin(s[8]) * sin(s[7]) - cos(s[8]) * cos(s[7]) * sin(s[6])) * u1
            v_dot[2]=-self.g + cos(s[7]) * cos(s[6]) * u1
            # v_dot[0]=(cos(yaw) * sin(pitch) + cos(pitch) * sin(roll) * sin(yaw)) * u1,
            # v_dot[1]=(sin(yaw) * sin(pitch) - cos(yaw) * cos(pitch) * sin(roll)) * u1,
            # v_dot[2]=self.g + cos(pitch) * cos(roll) * u1,
            q_dot=s[9:12]
            w_dot=torch.zeros(3)
            w_dot[0]=(self.Izz - self.Iyy) * s[10] * s[11]/ self.Ixx + u_roll / self.Ixx
            w_dot[1]=(self.Ixx - self.Izz) * s[9] * s[11]/ self.Iyy + u_pitch / self.Iyy
            w_dot[2]=(self.Iyy - self.Ixx) * s[9] * s[10]/ self.Izz + u_yaw / self.Izz
            # w_dot[0]=(self.Izz - self.Iyy) * pitch_rate * yaw_rate/ self.Ixx + u_roll / self.Ixx,
            # w_dot[1]=(self.Ixx - self.Izz) * roll_rate * yaw_rate/ self.Iyy + u_pitch / self.Iyy,
            # w_dot[2]=(self.Iyy - self.Ixx) * roll_rate * pitch_rate/ self.Izz + u_yaw / self.Izz,
            
            s_dot = torch.zeros((12,))
            s_dot[0:3] = x_dot
            s_dot[3:6] = v_dot
            s_dot[6:9] = q_dot
            s_dot[9:12] = w_dot
            return s_dot
        
        s = x.float()
        s=s+_s_dot_fn(self,s, u1,u_roll,u_pitch,u_yaw)*self.t_step
        return s
    
    def training_step(self, train_batch, batch_idx):
        # x,u,xt1,xt1_reality,x_dot=train_batch[:,:12],train_batch[:,12:16],train_batch[:,16:28],train_batch[:,28:40],train_batch[:,40:]
        x,u,xt1_reality=train_batch[:,:13],train_batch[:,13:17],train_batch[:,17:]
        # x,u,xt1_reality=train_batch[:,:12],train_batch[:,12:16],train_batch[:,16:]
        
        
        loss = torch.zeros(1).requires_grad_()

        # print(x.shape)
        # print(u.shape)

        # train_batch represents initial states
        num_traj = train_batch.shape[0]
        # y=torch.zeros((num_traj,12)).to(self.device)
        # error=torch.zeros((num_traj,12)).to(self.device)
        y=torch.zeros((num_traj,13)).to(self.device)
        error=torch.zeros((num_traj,13)).to(self.device)
        # print(num_traj)
        for i in range(num_traj):#
            y[i]=control_learner.xt1_ideal(self,x[i,:], u[i,:])+self.controller(x[i,:],u[i,:])*0.01
            # error=torch.zeros(1).to(self.device)
            # y[i]=x[i,:]+(x_dot[i,:]+self.controller(x[i,:],u[i,:]))*0.01# x_t+1=xt+(dot+controller)dt
            # y[i]=xt1
            error=xt1_reality[i,:]-y[i]#
            loss = loss + error.t()@error
        loss=loss*100
        # loss=loss
        self.log('train_loss', loss)
        # print("loss",loss)
        return loss
    
    def test_step(self, test_batch, batch_idx):
        x,u,xt1,xt1_reality=test_batch[:,:12],test_batch[:,12:16],test_batch[:,16:28],test_batch[:,28:]
    
        
        loss = torch.zeros(1).to(self.device)
        # print(x.shape)
        # print(u.shape)

        # train_batch represents initial states
        num_traj = test_batch.shape[0]
        y=torch.zeros((num_traj,12)).to(self.device)
        error=torch.zeros((num_traj,12)).to(self.device)
        # print(num_traj)
        for i in range(num_traj):
            # error=torch.zeros(1).to(self.device)
            y[i]=xt1[i,:]+self.controller(x[i,:],u[i,:])*0.01#13
            error=xt1_reality[i,:]-y[i]
            loss = loss + error.t()@error
        loss=loss/num_traj
        # loss=loss
        self.log('test_loss', loss)
        print("loss",loss)
        return loss

# if __name__=="__main__":
#     data=np.load("dataset.npy")
#     dataset=torch.tensor(data)

#     delta_nn_loader = DataLoader(dataset,batch_size = 10)      # mini batch size

#     model = control_learner()
#     # # training

#     trainer = pl.Trainer(accelerator="cpu", num_nodes=1,
#                         callbacks=[], max_epochs=2000)

#     trainer.fit(model, delta_nn_loader)
#     trainer.save_checkpoint("model_nonlinear_toy.ckpt")