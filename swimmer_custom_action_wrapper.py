import numpy as np 
from custom_swimmer import swimmer  # assuming swimmer.py is in custom_swimmer/
from wrapper import DMCWrapper
class SinusoidalFunc:
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
            return self.offset_odd+self.amplitude_odd*np.sin(theta+ self.delta)
        

class PIDController:
    def __init__(self, Kp, Kd, Ki =0.0):
        self.Kp = Kp
        self.Ki=Ki
        self.Kd = Kd
        self.integral = 0.0
        self.prev_error = 0.0

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self,desired_angle, actual_angle, dt ):
        error = desired_angle-actual_angle
        self.integral+=error*dt
        derivative = (error - self.prev_error)/dt
        self.prev_error=error

        return self.Kp*error + self.Ki*self.integral + self.Kd*derivative
    
class SwimmerCustomActionWrapper:
    def __init__(self, env, dt = 0.05):
        self.env = env 
        self.dt = dt
        self.t =0.0

        self.num_joints = self.env.action_space.shape[0]
        self.sinusoidal_func = SinusoidalFunc(self.num_joints)
        self.pid_controllers =[PIDController(Kp=1.0,Kd =1.0)for _ in range(self.num_joints)]


    def reset(self):
        self.t = 0.0
        timestep = self.env.reset()
        for pid in self.pid_controllers:
            pid.reset()
        return timestep
    
    def get_joint_angles(self):
        return self.env._env.physics.joints()
    def step(self, params): #params = [offset_even, amplitude_even, offset_odd, amplitude_odd, dtheta_dn, dtheta_dt, delta]
        self.sinusoidal_func.set_params(params)
        current_angles = self.get_joint_angles()
        torques = []
        
        for n in range(self.num_joints):
            desired = self.sinusoidal_func.get_desired_angle(n, self.t)
            torque = self.pid_controllers[n].compute(desired, current_angles[n], self.dt)
            torques.append(torque)


        """print(f"Torque Shape {np.array(torques).shape}")
        print(self.env.action_space.shape)
"""
        timestep = self.env.step(np.array(torques))
        self.t +=self.dt
        obs, reward, done, info = timestep

        print(info)
        return obs, reward, done
    

n_links = 12
dmc_env = swimmer.swimmer(n_links=n_links)

# 2. Wrap it as a Gym environment
gym_env = DMCWrapper(dmc_env)
custom_env = SwimmerCustomActionWrapper(gym_env)

obs = custom_env.reset()
for _ in range(10):
    params = np.random.uniform(-1,1,size = 7)
    obs, reward, done = (custom_env.step(params=params))
    """print(f"Obs is {obs}")

    print(f"reward is{reward}")

    print(f"done is {done}")"""