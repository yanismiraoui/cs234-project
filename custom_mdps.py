import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class RiverSwimEnv(gym.Env):
    def __init__(self, n_states=6, right_prob=0.4):
        super(RiverSwimEnv, self).__init__()
        self.n_states = n_states
        self.right_prob = right_prob
        self.n_actions = 2
        self.action_space = gym.spaces.Discrete(self.n_actions)
        self.observation_space = gym.spaces.Discrete(self.n_states)
        self.P = self._build_transition_probabilities()
        self.state = 0

    def _build_transition_probabilities(self):
        P = {}
        for s in range(self.n_states):
            P[s] = {a: [] for a in range(self.n_actions)}
        
        P[0][0] = [(1.0, 0, 0, False)]
        P[0][1] = [(1 - self.right_prob, 0, 0, False), (self.right_prob, 1, 0, False)]
        
        for s in range(1, self.n_states - 1):
            P[s][0] = [(1.0, s - 1, 0, False)]
            P[s][1] = [(1 - self.right_prob, s, 0, False), (self.right_prob, s + 1, 0, False)]
        
        P[self.n_states - 1][0] = [(1.0, self.n_states - 2, 0, False)]
        P[self.n_states - 1][1] = [(1 - self.right_prob, self.n_states - 1, 1, False), (self.right_prob, self.n_states - 1, 1, False)]
        
        return P

    def reset(self):
        self.state = 0
        return self.state

    def step(self, action):
        transitions = self.P[self.state][action]
        i = np.random.choice(len(transitions), p=[t[0] for t in transitions])
        prob, next_state, reward, done = transitions[i]
        self.state = next_state
        return next_state, reward, done, {}

    def render(self, mode='human'):
        pass

gym.envs.registration.register(
    id='RiverSwim-v0',
    entry_point='__main__:RiverSwimEnv',
)


class SixArmsEnv(gym.Env):
    def __init__(self, n_arms=6, transition_prob=0.2):
        super(SixArmsEnv, self).__init__()
        self.n_arms = n_arms
        self.transition_prob = transition_prob
        self.n_actions = n_arms 
        self.action_space = gym.spaces.Discrete(self.n_actions)
        self.observation_space = gym.spaces.Discrete(2)
        self.P = self._build_transition_probabilities()
        self.state = 0

    def _build_transition_probabilities(self):
        P = {0: {a: [] for a in range(self.n_actions)}, 1: {a: [] for a in range(self.n_actions)}}
        
        for a in range(self.n_actions):
            P[0][a] = [(self.transition_prob, 1, 1, False), (1 - self.transition_prob, 0, 0, False)]
        
        for a in range(self.n_actions):
            P[1][a] = [(1.0, 0, 0, False)]
        
        return P

    def reset(self):
        self.state = 0
        return self.state

    def step(self, action):
        transitions = self.P[self.state][action]
        i = np.random.choice(len(transitions), p=[t[0] for t in transitions])
        prob, next_state, reward, done = transitions[i]
        self.state = next_state
        return next_state, reward, done, {}

    def render(self, mode='human'):
        pass

gym.envs.registration.register(
    id='SixArms-v0',
    entry_point='__main__:SixArmsEnv',
)

class RandomMDPEnv(gym.Env):
    def __init__(self, n_states=5, n_actions=2, seed=None):
        super(RandomMDPEnv, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.action_space = gym.spaces.Discrete(self.n_actions)
        self.observation_space = gym.spaces.Discrete(self.n_states)
        
        if seed is not None:
            np.random.seed(seed)
        
        self.P = self._build_transition_probabilities()
        self.rewards = self._build_rewards()
        self.state = 0

    def _build_transition_probabilities(self):
        P = {}
        for s in range(self.n_states):
            P[s] = {a: [] for a in range(self.n_actions)}
            for a in range(self.n_actions):
                next_states = np.random.choice(self.n_states, self.n_states, replace=False)
                probabilities = np.random.dirichlet(np.ones(self.n_states))
                for prob, next_state in zip(probabilities, next_states):
                    P[s][a].append((prob, next_state, 0, False))
        return P

    def _build_rewards(self):
        rewards = np.random.uniform(-1, 1, (self.n_states, self.n_actions))
        return rewards

    def reset(self):
        self.state = 0
        return self.state

    def step(self, action):
        transitions = self.P[self.state][action]
        i = np.random.choice(len(transitions), p=[t[0] for t in transitions])
        prob, next_state, reward, done = transitions[i]
        reward = self.rewards[self.state, action]
        self.state = next_state
        return next_state, reward, done, {}

    def render(self, mode='human'):
        pass

# Register the custom environment
gym.envs.registration.register(
    id='RandomMDP-v0',
    entry_point='__main__:RandomMDPEnv',
)

class GarnetMDPEnv(gym.Env):
    def __init__(self, n_states=10, n_actions=3, sparsity=2):
        super(GarnetMDPEnv, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.sparsity = sparsity 
        self.action_space = gym.spaces.Discrete(self.n_actions)
        self.observation_space = gym.spaces.Discrete(self.n_states)
        self.P, self.R = self._build_transition_probabilities_and_rewards()
        self.state = 0

    def _build_transition_probabilities_and_rewards(self):
        P = {s: {a: [] for a in range(self.n_actions)} for s in range(self.n_states)}
        R = {s: {a: [] for a in range(self.n_actions)} for s in range(self.n_states)}
        
        for s in range(self.n_states):
            for a in range(self.n_actions):
                next_states = np.random.choice(self.n_states, self.sparsity, replace=False)
                probs = np.random.dirichlet(np.ones(self.sparsity))
                rewards = np.random.rand(self.sparsity)
                
                for next_state, prob, reward in zip(next_states, probs, rewards):
                    P[s][a].append((prob, next_state, reward, False))
                    R[s][a].append(reward)
        
        return P, R

    def reset(self):
        self.state = 0
        return self.state

    def step(self, action):
        transitions = self.P[self.state][action]
        i = np.random.choice(len(transitions), p=[t[0] for t in transitions])
        prob, next_state, reward, done = transitions[i]
        self.state = next_state
        return next_state, reward, done, {}

    def render(self, mode='human'):
        pass

gym.envs.registration.register(
    id='GarnetMDP-v0',
    entry_point='__main__:GarnetMDPEnv',
)