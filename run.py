
import ipdb
import matplotlib.pyplot as plt
from custom_swimmer import swimmer  # assuming swimmer.py is in custom_swimmer/
from wrapper import DMCWrapper, ActionMaskWrapper,SegmentActionWrapper,FrameSkipWrapper      # assuming you saved DMCWrapper in wrapper.py
from utils import pngs_to_gif

# 1. Create the raw dm_control environment with custom number of links
n_links = 12
dmc_env = swimmer.swimmer(n_links=n_links)

# 2. Wrap it as a Gym environment
gym_env = DMCWrapper(dmc_env)

# 3. Test: Reset and step through the environment
obs = gym_env.reset()
"""print("Initial obs shape:", obs.shape)

for step in range(5):
    action = gym_env.action_space.sample()  # random action
    obs, reward, done, info = gym_env.step(action)
    print(f"\nStep {step + 1}:")
    print("  Obs shape:", obs.shape)
    print("  Reward:", reward)
    print("  Done:", done)
    print("  Info:", info)
    if done:
        obs = gym_env.reset()"""

obs_spec = dmc_env.action_spec()
#print(obs_spec)
  # Your base wrapper
env = FrameSkipWrapper(SegmentActionWrapper(ActionMaskWrapper(gym_env,p=0),n_segments=4),skip=8)

obs = env.reset()
frames = []
for t in range(100):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    # ipdb.set_trace()
    img = env.env.env.env._env.physics.render()
    """ plt.imshow(img)
        # plt.show()
        plt.savefig('img'+str(t))
        frames.append('img'+str(t)+'.png')
        plt.close()"""
    print(f"\nStep {t + 1}:")
    print("  Obs shape:", obs.shape)
    print("  Reward:", reward)
    print("  Done:", done)
    print("  Info:", info)
    if done:
        obs = env.reset()

#pngs_to_gif(frames,output_path='gif')
