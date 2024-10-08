import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
import gymnasium as gym
from torch.distributions import Categorical

from A3C import A3C

MODEL_FOLDER = './models'

def main(epoch):

    for seed in range(0, 100):
        np.random.seed(seed)
        torch.manual_seed(seed)

        # env = gym.make('CartPole-v1', render_mode='human')
        env = gym.make('CartPole-v1')
        S_DIM = env.observation_space.shape[0]
        A_DIM = env.action_space.n
        a3c = A3C(is_central=False, s_dim=S_DIM, a_dim=A_DIM)
        a3c.actor.load_state_dict(torch.load(os.path.join(MODEL_FOLDER, 'actor_%d.pkl' % epoch)))

        s = env.reset()
        # env.render()
        s = s[0]
        done = False
        total_reward = 0
        while not done:
            s = torch.from_numpy(s).float()
            logits = a3c.actor(s).detach()
            probs = F.softmax(logits, dim=-1)
            m_probs = Categorical(probs)
            a = m_probs.sample().item()

            s_, r, done, truncated, _ = env.step(a)
            done = done or truncated
            s = s_
            total_reward += r
        
        print('Total reward:', total_reward)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=0)
    args = parser.parse_args()
    main(args.epoch)