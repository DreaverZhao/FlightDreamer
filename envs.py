import gymnasium as gym
import numpy as np
import torch

def getVecEnvProperties(env):
    assert isinstance(env.action_space, gym.spaces.Box), "Sorry, supporting only continuous action space for now"
    observationShape = env.observation_space.shape[1]
    actionSize = env.action_space.shape[1]
    # actionLow = env.action_space.low.tolist()[0]
    # actionHigh = env.action_space.high.tolist()[0]
    actionLow = torch.ones(actionSize, dtype=torch.float32) * -1
    actionHigh = torch.ones(actionSize, dtype=torch.float32) * 1
    return observationShape, actionSize, actionLow, actionHigh

def getEnvProperties(env):
    assert isinstance(env.action_space, gym.spaces.Box), "Sorry, supporting only continuous action space for now"
    observationShape = env.observation_space.shape
    actionSize = env.action_space.shape[0]
    actionLow = env.action_space.low.tolist()
    actionHigh = env.action_space.high.tolist()
    return observationShape, actionSize, actionLow, actionHigh

class GymPixelsProcessingWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        observationSpace = self.observation_space
        newObsShape = observationSpace.shape[-1:] + observationSpace.shape[:2]
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=newObsShape, dtype=np.float32)

    def observation(self, observation):
        observation = np.transpose(observation, (2, 0, 1))/255.0
        return observation
    
class CleanGymWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, reward, done

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        return obs
    
class IsaacGymWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = torch.bitwise_or(terminated, truncated)
        return obs["policy"], reward, done

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        return obs["policy"]