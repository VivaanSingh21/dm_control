import numpy as np 
from custom_swimmer import swimmer  # assuming swimmer.py is in custom_swimmer/
from wrapper import DMCWrapper, FrameSkipWrapper

import matplotlib.pyplot as plt
from utils import pngs_to_gif

"""class SinusoidalFunc:
    def __init__(self, num_joints):
        self.num_joints = num_joints
        self.set_params(np.zeros(7))

    def set_params(self, params):
        self.offset_even = params[0]
        self.amplitude_even = params[1]
        self.offset_odd = params[2]
        self.amplitude_odd = params[3]
        self.dtheta_dn = params[4]
        self.dtheta_dt = params[5]
        self.delta = params[6]

        #params = [offset_even, amplitude_even, offset_odd, amplitude_odd, dtheta_dn, dtheta_dt, delta]

    def calculate_theta(self, n, t):
        return self.dtheta_dn*n + self.dtheta_dt*t
    
    def get_desired_angle(self, n,t ):
        theta = self.calculate_theta(n,t)
        if n%2==0:
            return self.offset_even+self.amplitude_even*np.sin(theta)
        else:
            return self.offset_odd+self.amplitude_odd*np.sin(theta+ self.delta)"""
        

def sinusoid(params,t,n_joints):
    '''

    INPUTS:
    t is a scalar time
    n_joints is the nubmer of joints, and defines the dimensionality of the output
    '''

    offset, amp, dtheta_dn, dtheta_dt = params

    n = np.arange(n_joints) / (n_joints - 1)

    return offset + amp*np.sin(dtheta_dn*n + dtheta_dt*t)
        

class PIDController:
    def __init__(self, kp, ki, kd, dt = .05):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0
        self.dt = dt

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def __call__(self,desired_angle, actual_angle):
        error = desired_angle-actual_angle
        # print('desired angle ', desired_angle)
        # print(f"actual angle {actual_angle}")
        #print(f"Config oif PID COntroller {self.kp, self.kd, self.ki}")
        self.integral+=error*self.dt
        derivative = (error - self.prev_error)/self.dt
        self.prev_error=error

        return self.kp*error + self.ki*self.integral + self.kd*derivative  # this is correct, assuming the joint torque INCREASES the angle
    
class SwimmerCustomActionWrapper:
    def __init__(self, env, dt = 0.05):
        self.env = env 
        self.dt = dt
        self.t = 0.0

        self.num_joints = self.env.action_space.shape[0]
        # self.sinusoidal_func = SinusoidalFunc(self.num_joints)
        # self.pid_controllers = [PIDController(Kp=1.0,Kd =1.0)for _ in range(self.num_joints)]
        self.pid_controller = PIDController(kp=1.,kd=1.,ki=0.,dt=self.dt)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.reward_range = self.env.reward_range
        self.metadata = self.env.metadata
        self.logged_data = []


    def reset(self):
        self.t = 0.0
        timestep = self.env.reset()
        self.pid_controller.reset()
        # for pid in self.pid_controllers:
        #     pid.reset()
        return timestep
    
    def get_pid_config(self):
        return (self.pid_controller.kp, self.pid_controller.ki, self.pid_controller.kd)
    
    def get_joint_angles(self):
        return self.env._env.physics.joints()
    
    def step(self, action):#action =[offset, amp, dtheta_dn, dtheta_dt]
        # self.sinusoidal_func.set_params(params)
        
        desired_angles = sinusoid(action,self.t,self.num_joints)
       
        current_angles = self.get_joint_angles()
        
        

        torques = self.pid_controller(desired_angles,current_angles)
        torques =np.clip(torques,-1,1)
        

        self.logged_data.append((self.t, desired_angles.copy(), current_angles.copy(), torques.copy()))
        # torques = []
        
        # for n in range(self.num_joints):
        #     desired = self.sinusoidal_func.get_desired_angle(n, self.t)
        #     torque = self.pid_controllers[n].compute(desired, current_angles[n], self.dt)
        #     torques.append(torque)


        """print(f"Torque Shape {np.array(torques).shape}")
        print(self.env.action_space.shape)
"""
        timestep = self.env.step(torques)
        self.t += self.dt
        obs, reward, done, info = timestep

        #print(info)
        
        return obs, reward, done, info
    

    def plot_joint_graphs(self):
        ts, desireds, actuals, torques = zip(*self.logged_data)
        ts = np.array(ts)
        desireds = np.stack(desireds)  # shape: (T, n_joints)
        actuals = np.stack(actuals)    # shape: (T, n_joints)
        torques = np.stack(torques)
        print(np.max(desireds))

        n_joints = desireds.shape[1]
        for j in range(n_joints):
            plt.figure()
            plt.plot(ts, torques[:, j], label=f'Torque Joint {j}')
            plt.plot(ts, desireds[:, j], label=f'Desired Joint {j}')
            plt.plot(ts, actuals[:, j], label=f'Actual Joint {j}', linestyle='--')
            plt.xlabel("Time (s)")
            plt.ylabel("Joint Angle (rad)")
            plt.title(f"Joint {j}: Desired vs. Actual - {self.get_pid_config()}")
            plt.legend()
            plt.grid(True)
            #plt.show()
            plt.savefig(f'joint_{j}_tracking.png')
            
            plt.close()

        # Average across all joints per timestep
        # mean_desired = np.mean(desireds, axis=1)  # shape (T,)
        # mean_actual = np.mean(actuals, axis=1)
        # mean_torques = np.mean(torques, axis=1)     # shape (T,)

        # plt.figure(figsize=(5, 5))
        # plt.plot(ts, mean_desired, label='Mean Desired Angle', linestyle='-')
        # plt.plot(ts, mean_actual, label='Mean Actual Angle', linestyle='--')
        # plt.plot(ts, mean_torques, label='Mean Torque Angle', linestyle='--')
        # plt.xlabel("Time (s)")
        # plt.ylabel("Mean Joint Angle (rad)")
        # plt.title("Mean Joint Angle Over Time")
        # plt.legend()
        # plt.grid(True)
        # plt.tight_layout()
        # plt.xlim(0,10)
        # plt.savefig("joint_mean_angle_tracking.png")
        # plt.close()

    def analyze_step_response(self, joint_idx=0):
        if not self.logged_data:
            print("No data logged yet.")
            return

        ts, desireds, actuals, torques = zip(*self.logged_data)
        ts = np.array(ts)
        desireds = np.stack(desireds)[:, joint_idx]
        actuals = np.stack(actuals)[:, joint_idx]

        final_value = desireds[-1]
        response = actuals

        # Rise Time: Time to go from 10% to 90% of final value
        rise_start = final_value * 0.1
        rise_end = final_value * 0.9
        rise_mask = (response >= rise_start) & (response <= rise_end)
        rise_indices = np.where(rise_mask)[0]
        if len(rise_indices) > 0:
            rise_time = ts[rise_indices[-1]] - ts[rise_indices[0]]
        else:
            rise_time = None

        # Overshoot: Max value beyond final, as % of final
        peak = np.max(response)
        overshoot = ((peak - final_value) / final_value) * 100 if peak > final_value else 0

        # Plotting
        plt.figure(figsize=(8, 5))
        plt.plot(ts, desireds, label="Step Input", linestyle='--')
        plt.plot(ts, actuals, label=f"Actual Joint {joint_idx}", linewidth=2)
        plt.xlabel("Time (s)")
        plt.ylabel("Joint Angle (rad)")
        plt.title(f"Step Response: Joint {joint_idx} - {self.get_pid_config()}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"step_response_joint_{joint_idx}.png")
        plt.close()

        # Print results
        print(f"\nStep Response Analysis for Joint {joint_idx}:")
        print(f"↳ Rise time: {rise_time:.3f} s" if rise_time else "↳ Rise time: Not detected")
        print(f"↳ Overshoot: {overshoot:.2f} %")



     
    

n_links = 51
dmc_env = swimmer.swimmer(n_links=n_links)

# 2. Wrap it as a Gym environment
gym_env = DMCWrapper(dmc_env)
custom_env = FrameSkipWrapper(SwimmerCustomActionWrapper(gym_env), skip =4)
custom_env.env.pid_controller = PIDController(1,0.5,0)

frames = []
obs = custom_env.reset()
for t in range(500):
    print(t)
    params = np.array([0,7.5,1,1])
    obs, reward, done, info = (custom_env.step(action=params))
    img = custom_env.env.env._env.physics.render()

    plt.imshow(img)
        # plt.show()
    plt.savefig('img'+str(t))
    frames.append('img'+str(t)+'.png')
    plt.close()
    # print(f"Obs is {obs}")

    # print(f"reward is{reward}")

    print(f"done is {done}")


pngs_to_gif(frames,output_path='gif')
#custom_env.plot_joint_graphs()

# for i in range(custom_env.num_joints):
    
#     custom_env.env.analyze_step_response(joint_idx=i)

# custom_env.env.analyze_step_response_average()













"Below here is the step response experiment"






# n_links = 7
# dmc_env = swimmer.swimmer(n_links=n_links)
# custom_env = DMCWrapper(dmc_env)
# custom_env = FrameSkipWrapper(custom_env, skip=4)
# pid_controller = PIDController(1,0.5,0)
# actual_angle_lst =[]
# desired_angle_lst = []
# actions_lst = []
# custom_env.reset()
# for i in range(1000):
    
#     actual_angle = custom_env.env._env.physics.joints()
#     if i<500:
#         desired_angles = np.zeros(custom_env.action_space.shape[0])
#         actions = np.zeros(custom_env.action_space.shape[0])
#     else:
#         desired_angles = np.ones(custom_env.action_space.shape[0])
#         actions = pid_controller(desired_angles, actual_angle)
#     actions = np.clip(actions,-1,1)
#     print(actions)
#     actual_angle_lst.append(actual_angle)
#     desired_angle_lst.append(desired_angles)
#     actions_lst.append(actions)

#     custom_env.step(actions)

#     img = custom_env.env._env.physics.render()

    # plt.imshow(img)
    #     # plt.show()
    # plt.savefig('img'+str(t))
    # frames.append('img'+str(t)+'.png')
    # plt.close()
    # print(f"Obs is {obs}")

    # print(f"reward is{reward}")

    # print(f"done is {done}")


#pngs_to_gif(frames,output_path='gif')
#custom_env.plot_joint_graphs()
    

# action_stacked = np.stack(actions_lst)
# desired_stack = np.stack(desired_angle_lst)
# actual_angle_stack = np.stack(actual_angle_lst)

# for i in range(action_stacked.shape[1]):


#     plt.figure(figsize=(8, 5))
#     plt.plot( desired_stack[:, i], label=f"Desired angle joint{i}")
#     plt.plot( action_stacked[:, i], label=f"Action(Torque) joint {i}")
#     plt.plot( actual_angle_stack[:, i], label=f"actual angle joint {i}")
#     plt.xlabel("Time (s)")
#     plt.ylabel("Joint Angle (rad)")
#     plt.title(f"Step Response Graph")
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(f"step_response{i}.png")
#     plt.close()