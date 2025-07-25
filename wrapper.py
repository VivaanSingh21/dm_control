import gym
from gym import spaces
import numpy as np
import ipdb
class DMCWrapper(gym.Env):
    def __init__(self, dmc_env):
        self._env = dmc_env
        obs_spec = self._env.observation_spec()
        action_spec = self._env.action_spec()

        # Flatten dict observation
        obs_dim = sum(np.prod(v.shape) for v in obs_spec.values())
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=action_spec.minimum,
            high=action_spec.maximum,
            dtype=np.float32
        )

    def reset(self):
        ts = self._env.reset()
        self._last_time_step = ts  # Save for info dict
        return self._flatten_obs(ts.observation)

    def step(self, action):
        ts = self._env.step(action)
        self._last_time_step = ts  # Save for info dict

        reward = ts.reward or 0.0
        done = ts.last()
        obs = self._flatten_obs(ts.observation)

        info = self._build_info(ts)

        return obs, reward, done, info

    def _flatten_obs(self, obs_dict):
        return np.concatenate([v.ravel() for v in obs_dict.values()])

    def _build_info(self, ts):
        physics = self._env.physics
        info = {}

        # Example info fields you might want:
        try:
            info['to_target_distance'] = physics.nose_to_target_dist()
            info['to_target_vector'] = physics.nose_to_target()
            info['joint_angles'] = physics.joints()
            info['body_velocities'] = physics.body_velocities()
        except Exception as e:
            info['warning'] = f"Could not build some info fields: {str(e)}"

        return info


class ActionMaskWrapper(gym.Wrapper):
    def __init__(self, env,p):
        super().__init__(env)
        '''
        p is the probability that a link is disabled
        '''
        self.env = env
        self.action_space = env.action_space
        self.a_dim = self.action_space.shape[0]
        self.mask = None
        self.p = p
    def reset(self):
        obs = self.env.reset()
        # self.mask = np.random.uniform(low =-1.0, high = 1.0, size = self.action_space.shape)
        self.mask = np.random.binomial(n=1,p=1-self.p,size=self.a_dim)
        return obs
    def step(self,action):
        assert self.mask is not None #Ensures that step is not called before reset
        masked_action = self.mask* action
        obs,reward,done,info = self.env.step(masked_action)
        info["action_mask"] = self.mask
        return obs, reward, done, info 
    

class SegmentActionWrapper(gym.Wrapper):
    def __init__(self, env, n_segments):
        super().__init__(env)
        self.env = env
        self.n_segments = n_segments

        # Total number of joints (original action space dimension)
        self.action_dim = self.env.action_space.shape[0]

        # Compute segment boundaries
        base = self.action_dim // n_segments
        remainder = self.action_dim % n_segments
        self.segment_sizes = [base + 1 if i < remainder else base for i in range(n_segments)]

        # New lower-dimensional action space
        low = np.full((n_segments,), self.env.action_space.low.min())
        high = np.full((n_segments,), self.env.action_space.high.max())
        self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def get_low_level_actions(self, action):
        '''
        Expand a low-dimensional action into the full action space
        by repeating each segment action value over its corresponding joints.
        '''
        assert len(action) == self.n_segments, "Action size must match number of segments"
        full_action = []
        for i, size in enumerate(self.segment_sizes):
            full_action.extend([action[i]] * size)
        return np.array(full_action, dtype=np.float32)

    def step(self, action):
        #print('action: ', action)
        full_action = self.get_low_level_actions(action)
        #print('full_action: ', full_action)
        return self.env.step(full_action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class FrameSkipWrapper(gym.Wrapper):
    def __init__(self, env, skip=4):
        """
        Repeat the same action for `skip` frames.

        Args:
            env (gym.Env): Environment to wrap.
            skip (int): Number of times to repeat each action.
        """
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        info = {}
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

def sinusoid(params,t,n_joints):
    '''

    INPUTS:
    t is a scalar time
    n_joints is the nubmer of joints, and defines the dimensionality of the output
    '''
    offset, amp, dtheta_dn, dtheta_dt = params

    n = np.arange(n_joints) # / (n_joints - 1)

    return offset + amp*np.sin(dtheta_dn*n + dtheta_dt*t)
   
        

class PIDController:
    def __init__(self, kp, ki, kd, min_output =-1, max_output = 1, dt = .05):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0
        self.dt = dt
        self.min_output = min_output
        self.max_output = max_output

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def update_integrator(self,error,output):
        # ((x > max_output) | (x < min_output)).astype(np.float64)
        if self.min_output is not None and self.max_output is not None:
            integrator_mask = ((output > self.max_output) | (output < self.min_output)).astype(np.float64)

            return self.integral + integrator_mask*error*self.dt

        return self.integral + error*self.dt

    def __call__(self,desired_angle, actual_angle):
        error = desired_angle-actual_angle
        # print(f"error is {error}")
        # print('desired angle ', desired_angle)
        # print(f"actual angle {actual_angle}")
        #print(f"Config oif PID COntroller {self.kp, self.kd, self.ki}")
        
        derivative = (error - self.prev_error)/self.dt
        self.prev_error=error

        output = self.kp*error + self.ki*self.integral + self.kd*derivative  # this is correct, assuming the joint torque INCREASES the angle

        self.integral = self.update_integrator(error,output)
        
        
        
        
        return output





    
class SwimmerCustomActionWrapper(gym.Wrapper):
    def __init__(self, env, action_scale=np.array([1.,1.,np.pi,np.pi/.05]), dt = 0.05):
        super().__init__(env) #We are having to make this class a proper subclass of gym.Wrapper to enable compatibality with stable baseline 3
        self.env = env 
        self.dt = dt
        self.t = 0.0

        self.num_joints = self.env.action_space.shape[0]
        # self.sinusoidal_func = SinusoidalFunc(self.num_joints)
        # self.pid_controllers = [PIDController(Kp=1.0,Kd =1.0)for _ in range(self.num_joints)]
        self.pid_controller = PIDController(kp=1., ki=.5, kd= 0,dt=self.dt)
        # self.action_space = self.env.action_space
        
        a_dim = 4  # 4-dimensional actions for offset, amplitude, dtheta/dn, dtheta/dt
        self.action_scale = action_scale  # this is for if we want actions of +/-1 to map to larger/smaller sinusoid parameters
        assert self.action_scale.shape[0] == a_dim

        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(a_dim,),
            dtype=np.float32
        )
        self.observation_space = self.env.observation_space
        self.reward_range = self.env.reward_range
        self.metadata = self.env.metadata




    def reset(self):
        self.t = 0.0
        timestep = self.env.reset()
        self.pid_controller.reset()
        # for pid in self.pid_controllers:
        #     pid.reset()
        return timestep
   
    def get_joint_angles(self):
        return self.env._env.physics.joints()
    
    def step(self, action):#action =[offset, amp, dtheta_dn, dtheta_dt]
        
        "Regular  funcion with high level action entering sinusoid to yield desired angle"
        action = self.action_scale * action

        desired_angles = sinusoid(action,self.t,self.num_joints)
        # print(f"Desired Angles:{desired_angles}")
        current_angles = self.get_joint_angles()
        # print(f"Current angles: {current_angles}")
        # ipdb.set_trace()
        torques = self.pid_controller(desired_angles,current_angles)
        #print(f"Torques: {torques}")
        # print(torques)
        torques = np.clip(torques,-1,1)
        

        obs, reward, done, info = self.env.step(torques)
        self.t += self.dt


        return obs, reward, done, info

        "Test  function with high-level action entering sinusoid to yield torque"
        # action = self.action_scale * action

        # torques = sinusoid(action,self.t,self.num_joints)
       
        # print(f"Torques: {torques}")
        # # print(torques)
        # torques = np.clip(torques,-1,1)
        

        # obs, reward, done, info = self.env.step(torques)
        # self.t += self.dt

        
        # self.torques.append(torques)
        
        # info = {
        #         'torques':np.stack(self.torques)}
        # return obs, reward, done, info

    


    
class SwimmerCustomActionWrapperTorque(gym.Wrapper):
    def __init__(self, env, action_scale=np.array([.3,1.,.5*np.pi,.3*np.pi/.05]), dt = 0.05):
        super().__init__(env) #We are having to make this class a proper subclass of gym.Wrapper to enable compatibality with stable baseline 3
        self.env = env 
        self.dt = dt
        self.t = 0.0

        self.num_joints = self.env.action_space.shape[0]
        # self.sinusoidal_func = SinusoidalFunc(self.num_joints)
        # self.pid_controllers = [PIDController(Kp=1.0,Kd =1.0)for _ in range(self.num_joints)]
        
        # self.action_space = self.env.action_space
        
        a_dim = 4  # 4-dimensional actions for offset, amplitude, dtheta/dn, dtheta/dt
        self.action_scale = action_scale  # this is for if we want actions of +/-1 to map to larger/smaller sinusoid parameters
        assert self.action_scale.shape[0] == a_dim

        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(a_dim,),
            dtype=np.float32
        )
        self.observation_space = self.env.observation_space
        self.reward_range = self.env.reward_range
        self.metadata = self.env.metadata


    def reset(self):
        self.t = 0.0
        timestep = self.env.reset()
       
        # for pid in self.pid_controllers:
        #     pid.reset()
        return timestep
   
    def get_joint_angles(self):
        return self.env._env.physics.joints()
    
    def step(self, action):#action =[offset, amp, dtheta_dn, dtheta_dt]
        


        "Test  function with high-level action entering sinusoid to yield torque"
        action = self.action_scale * action

        torques = sinusoid(action,self.t,self.num_joints)
       
        # print(f"Torques: {torques}")
        # print(torques)
        torques = np.clip(torques,-1,1)
        

        obs, reward, done, info = self.env.step(torques)
        self.t += self.dt

        

        

        return obs, reward, done, info




forward_A = 1
backward_A = .25

gates = [
(0,forward_A,1,-1), #Forward
(.5,forward_A,1,-1), #Forward Right Tight
(.2,forward_A,1,-1), #Forward Right Loose
(-0.5,forward_A,1,-1), #Forward Left Tight
(-0.2,forward_A,1,-1), #Forward Left Loose
(0,backward_A,1,1), #Backward
(-0.5,backward_A,1,1), #Backward Right Tight
(-0.2,backward_A,1,1), #Backward Right Loose
(0.5,backward_A,1,1), #Backward Left Tight
(0.2,backward_A,1,1) #Backward Left Loose
]

class OneHotActionSpace:
    def __init__(self,a_dim):
        self.a_dim = a_dim
        self.shape = (self.a_dim,)
    def sample(self):
        index = np.random.randint(self.a_dim)
        one_hot = np.zeros(self.a_dim)
        one_hot[index]=1
        return one_hot
    
class GateWrapper(gym.Wrapper):
    def __init__(self, env, gate_encoding=gates):
        super().__init__(env)

        self.env = env
        self.gate_encoding = gate_encoding
        a_dim = len(gate_encoding) #1-hot encoded vector for our 10 gates
       
        # self.action_space = gym.spaces.MultiBinary(a_dim)
        self.action_space = OneHotActionSpace(a_dim) #gym.spaces.Discrete(a_dim)

        self.observation_space = env.observation_space

    def reset(self):
        self.env.reset()

    def step(self, action):
        assert (np.sum(action)==1 and len(action)== len(self.gate_encoding)), "Action must be a valid 1 hot vector"
        gate_index = np.argmax(action)
        return self.env.step(np.array(self.gate_encoding[gate_index]))