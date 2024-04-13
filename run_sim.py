from cProfile import label
import quadrotor
import controller
import trajectory
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation
from utils import *
import time
import math
from scipy.spatial.transform import Rotation
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from delta_nn import *
from mpc_control_dob import instantiate_controller
from observer import Observer

# start simulation
if __name__=="__main__":
    # dataset=[]
    # # data=np.load("circle_mass0.04_13.npy")
    # data=np.load("R4circle_mass0.05_13.npy")
    # dataset=torch.tensor(data)

    # delta_nn_loader = DataLoader(dataset[:2000],batch_size = 100, shuffle=True)      # mini batch size
    # # delta_nn_loader_test = DataLoader(dataset[2000:],batch_size = 500, shuffle=True)
    # seed=2
    # pl.seed_everything(seed=seed)
    # model = control_learner()
    # # training

    # # trainer = pl.Trainer(accelerator="cpu", num_nodes=1,
    # #                     callbacks=[], max_epochs=1000)
    # trainer = pl.Trainer(accelerator="cpu", num_nodes=1,
    #                     callbacks=[], max_epochs=1000)

    # trainer.fit(model, delta_nn_loader)
    # # trainer.save_checkpoint("masschange_model_knode_1000_13_dia.ckpt")
    # trainer.save_checkpoint("R4masschange0.04circle_my_model_1000_13_dia.ckpt")

    # # new_model = control_learner.load_from_checkpoint(
    # #         checkpoint_path="masschange_model_knode_1000_13_dia.ckpt")
    # new_model = control_learner.load_from_checkpoint(
    #         checkpoint_path="R4masschange0.04circle_my_model_1000_13_dia.ckpt")
    
    # # trainer.test(new_model,delta_nn_loader_test)

    # # torch.save (new_model.state_dict (), "masschange_my_model_1000_13_dia.pth")
    # torch.save (new_model.state_dict (), "R4masschange0.04circle_my_model_1000_13_dia.pth")



    T_block=np.array([20,16,18,22,24])         ###########
    radius_block=np.array([7,6,5,4,3,2,1]) #############
    for T in T_block:
        for radius in radius_block:
            
            observer= Observer()
            quad_model = quadrotor.Quadrotor()

            quad_model.reset()
            real_trajectory = {'x': [], 'y': [], 'z': []}
            des_trajectory = {'x': [], 'y': [], 'z': []}
            
            accu_error_pos = np.zeros((3, ))
            total_time = 0
            square_ang_vel = np.zeros((4, ))
            
            simu_freq = 100 # Hz
            ctrl_freq = 50
            traj = trajectory.Trajectory("circle",T,radius)
            # traj = trajectory.Trajectory("diamond",T,radius)
            dataset=[]
            # quad_controller = controller.Linear_MPC(traj, ctrl_freq, use_obsv=False)
                            
            # quad_controller = controller.NonLinear_MPC(traj, ctrl_freq)
            quad_controller=instantiate_controller()

            # quad_controller = controller.PDcontroller(traj, ctrl_freq)

            simu_time = 20 # sec
            
            cur_time = 0
            dt = 1 / simu_freq
            num_iter = int(simu_time * simu_freq)
            start = time.time()

            x_t1=np.zeros((12))
            x_t13=np.zeros((13))
            x_t=np.zeros((12))
            x_dot=np.zeros((12))
            x_td12=np.zeros((12))
            state_real=np.zeros((13))
            x_td13=np.zeros((13))

            u_obs=np.zeros((1,3))####DOB
            g = 9.81  # m/s^2
            arm_length=0.046
            A=np.array([[0,1/(2*math.pi)*np.sqrt(g/arm_length),0,0,0,0],
                                [-1/(2*math.pi)*np.sqrt(g/arm_length),0,0,0,0,0],
                                [0,0,0,1/(2*math.pi)*np.sqrt(g/arm_length),0,0],
                                [0,0,-1/(2*math.pi)*np.sqrt(g/arm_length),0,0,0],
                                [0,0,0,0,0,1/(2*math.pi)*np.sqrt(g/arm_length)],
                                [0,0,0,0,-1/(2*math.pi)*np.sqrt(g/arm_length),0]])
            C=np.array([[1, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0]])
            xi=np.array([[1,0,0,0,0,0]])

            # u_obs_ESO=np.zeros((1,3))####ESO
            u_obs_DOB=np.zeros(3)####DOB
            u_obs_ESO=np.zeros(3)####ESO
            u_obs_ESOro=np.zeros(3)##ESOro
            yita=np.zeros(3)
            # u_dis=np.zeros((1,1))####ESO
            obs_rotate=np.zeros((1,1))

            for i in range(num_iter):
                des_state = traj.get_des_state(cur_time)

                ####### DOB #######
                xi_dot=(A@xi.T).T
                xi=xi+xi_dot*dt
                dis=(C@xi.T).T
                # print(dis)
                

                x_t13[0:3] = quad_model.state['x'].squeeze()
                x_t13[3:6] = quad_model.state['v'].squeeze()
                x_t13[6:10] = quad_model.state['q'].squeeze()
                x_t13[10:13] = quad_model.state['w'].squeeze()

                if i % (int(simu_freq / ctrl_freq)) == 0:

                    # control_input, error_pos = quad_controller.control(cur_time, quad_model.state)####DOB
                    # control_input= quad_controller.update( quad_model.state,des_state,u_dis=dis,obs_compensate_DOB=u_obs_DOB[-1])####DOB
                    # control_input= quad_controller.update( quad_model.state,des_state,obs_compensate_ESO=u_obs_ESO)####ESO
                    control_input,error_pos= quad_controller.update(quad_model.state,des_state,u_dis=dis,obs_compensate_DOB=u_obs_DOB,obs_compensate_ESO=u_obs_ESO,obs_compensate_ro=u_obs_ESOro,cur_time=cur_time)####DOB+ESO
                    
                    
                cmd_rotor_speeds = control_input["cmd_motor_speeds"]

                # obs, _, _, _ ,ades= quad_model.step(cmd_rotor_speeds) #x(t+1)=f(x,u)+nn(xt,u)
                obs, _, _, _,ades = quad_model.step_real(cmd_rotor_speeds)#x(t+2)=f(xt+1,u)+g(xt+1,u)
                

                x_td13[0:3] = obs['x'].squeeze()
                x_td13[3:6] = obs['v'].squeeze()
                x_td13[6:10] = obs['q'].squeeze()
                x_td13[10:13] = obs['w'].squeeze()
                # x_dot=(x_t1-x_t)/0.01

                observer.observerDOB(control_input["cmd_desire"][0],dt,x_t13[6:10],x_t13[3:6],ades,x_t13,control_input["cmd_desire"],u_obs_ESO)###DOB
                u_obs_DOB=np.array(observer.distDOB[-1])
                # print("obs_DOB",u_obs_DOB)
                observer.observerESO(u1=control_input["cmd_desire"][0],dt=dt,stateQ=x_t13[6:10],xp=x_t13[0:3],state13=x_t13,u4=control_input["cmd_desire"],u_obs_DOB=u_obs_DOB)##ESO
                u_obs_ESO=np.array(observer.distESO[-1])###ESO
                # print("obs_ESO",u_obs_ESO)

                yita=quad_controller.euler_from_quaternion(x_t13[6],x_t13[7],x_t13[8],x_t13[9])#phi,theta,psi
                observer.observerESOro(control_input["cmd_desire"][1:],dt,x_t13[6:10],yita,x_t13,control_input["cmd_desire"])###ESOro 
                u_obs_ESOro=np.array(observer.distESOro[-1])
                # print("obs_ESOro",u_obs_ESOro)

                cur_time += dt

                ##数据集构建
                data=np.concatenate((x_t13,control_input["cmd_desire"],x_td13),axis=0)
                dataset.append(data)

                accu_error_pos += error_pos**2 * dt
                square_ang_vel += cmd_rotor_speeds ** 2 * dt
                
                real_trajectory['x'].append(obs['x'][0])
                real_trajectory['y'].append(obs['x'][1])
                real_trajectory['z'].append(obs['x'][2])

                
                des_trajectory['x'].append(des_state['x'][0][0])
                des_trajectory['y'].append(des_state['x'][1][0])
                des_trajectory['z'].append(des_state['x'][2][0])

            tracking_performance=np.sum(accu_error_pos)
            '''Print three required criterions'''
            print("T= ",T)
            print("radius=",radius)
            print("Tracking performance: ", np.sum(accu_error_pos))
            # np.savetxt('Tracking performance_OB:',np.sum(accu_error_pos**2).reshape(1,-1))
            # print("Sum of square of angular velocities: ", np.sum(square_ang_vel))
            # print("Total time: ", time.time() - start)
            ##数据集构建
            # np.save("R4circle_mass0.05_13.npy", np.array(dataset))

            '''Visualization'''
            visualizer = Visualizer(simu_time, simu_freq, ctrl_freq, real_trajectory, des_trajectory)
            visualizer.plot_tracking_performance(T,radius,tracking_performance)
            # visualizer.record_tracking_data()

            # try:
            #     if quad_controller.use_obsv == True:
            #         visualizer.plot_obsv_x(quad_controller.x_real, quad_controller.x_obsv)
            #         # visualizer.plot_obsv_d(quad_controller.d_hat_list)
            # except:
            #     pass
            # visualizer.animation_3d()
    
