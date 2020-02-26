"""
Record trajectory data with the DataCollectionWrapper wrapper and play them back.

Example:
    $ python demo_collect_and_playback_data.py --environment BaxterLift
"""

import os
import argparse
from glob import glob
import numpy as np

import robosuite
#from robosuite import DataCollectionWrapper
#from robosuite.wrappers import IKWrapper
from robosuite.wrappers import GymWrapper

def collect_random_trajectory(env, timesteps=1000):
    """Run a random policy to collect trajectories.

    The rollout trajectory is saved to files in npz format.
    Modify the DataCollectionWrapper wrapper to add new fields or change data formats.
    """

    obs = env.reset()
    dof = env.dof

    for t in range(timesteps):
        action = 0.5 * np.random.randn(dof)
        obs, reward, done, info = env.step(action)
        env.render()
        if t % 100 == 0:
            print(t)


def playback_trajectory(env, ep_dir):
    """Playback data from an episode.

    Args:
        ep_dir: The path to the directory containing data for an episode.
    """

    # first reload the model from the xml
    #xml_path = os.path.join(ep_dir, "model.xml")
    #with open(xml_path, "r") as f:
    #    env.reset_from_xml_string(f.read())

    # fOR MODELS
    #state_paths = os.path.join(ep_dir, "model_*.npz") # change from state to model
    
    # FOr combined
    state_paths = os.path.join(ep_dir, "combined_*.npz") # change from state to model

    # read states back, load them one by one, and render
    t = 0
    for state_file in sorted(glob(state_paths)):
        print(state_file)
        dic = np.load(state_file)
        
        # FOr MOdels
        #states = dic["sims"] # sims observation instead states
        #actions = dic["acs"]

        # For Combined
        #states = dic["sims"][0] # sims observation instead states
        actions = dic["acs"]

        #print("shape of actions: ", actions.shape)
        
        # Play from action
        #env.sim.set_state_from_flattened(states[0])

        # For combined
        env.reset()
        env.sim.forward()

        #print("environment action space: ", env.action_spec) # check out action space
        #print("an actions shape: ", actions[0].shape) # notics take the tensor out
        #print("environment degree of freedom: ", env.dof)

        for action in actions[3]:
            #print("action length: ", len(action))
            env.step(action)
            env.render()

        # Play from action instead of states for testing
        #for state in states:
        #    env.sim.set_state_from_flattened(state)
        #    env.sim.forward()
        #    env.render()
        #    t += 1
        #    if t % 100 == 0:
        #        print(t)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="SawyerLift") # Change from sawyerstack to sawyerlift for playback of specific case

    # For models
    #parser.add_argument("--directory", type=str, default="/home/mastercljohnson/Robotics/GAIL_Part/mod_surreal/robosuite/models/assets/demonstrations/ac100/models/")

    # For checking combined
    parser.add_argument("--directory", type=str, default="/home/mastercljohnson/Robotics/GAIL_Part/mod_surreal/robosuite/models/assets/demonstrations/ac100/combined/")

    parser.add_argument("--timesteps", type=int, default=2000)
    args = parser.parse_args()

    # create original environment
    env = robosuite.make(
        args.environment,
        ignore_done=True,
        use_camera_obs=False,
        has_renderer=True,
        control_freq=100,
    )
    data_directory = args.directory

    # wrap the environment with data collection wrapper
    #env = DataCollectionWrapper(env, data_directory)
    #env = IKWrapper(env)
    env = GymWrapper(env)

    # testing to make sure multiple env.reset calls don't create multiple directories
    env.reset()
    env.reset()
    env.reset()

    # collect some data
    #print("Collecting some random data...")
    #collect_random_trajectory(env, timesteps=args.timesteps)

    # playback some data
    _ = input("Press any key to begin the playback...")
    print("Playing back the data...")
    #data_directory = env.ep_directory
    playback_trajectory(env, data_directory)
