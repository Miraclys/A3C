import os
import logging
import argparse
import torch.multiprocessing as mp
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from datetime import datetime
from tensorboardX import SummaryWriter
import gymnasium as gym
from torch.distributions import Categorical

from A3C import A3C

A_DIM = 6
ACTOR_LR_RATE = 0.01
CRITIC_LR_RATE = 0.01
NUM_AGENTS = 8
TRAIN_SEQ_LEN = 100
RANDOM_SEED = 42
MODEL_SAVE_INTERVAL = 50
EPOCH = 100000

MODEL_FOLDER = './models'
TEST_RESULTS = './test_results'


def eval_model(epoch):

    env = gym.make('CartPole-v1')
    S_DIM = env.observation_space.shape[0]
    A_DIM = env.action_space.n
    a3c = A3C(is_central=False, s_dim=S_DIM, a_dim=A_DIM)
    a3c.actor.load_state_dict(torch.load(os.path.join(MODEL_FOLDER, 'actor_%d.pkl' % epoch)))

    s, _ = env.reset()
    done = False
    total_reward = []
    while not done:
        s = torch.from_numpy(s).float()
        logits = a3c.actor(s).detach()
        probs = F.softmax(logits, dim=-1)
        m_probs = Categorical(probs)
        a = m_probs.sample().item()

        s_, r, done, truncated, info = env.step(a)
        done = done or truncated
        s = s_
        total_reward.append(r)

    return np.sum(total_reward)


def central_agent(pre_epoch, net_param_queues, exp_queues):
    assert len(net_param_queues) == NUM_AGENTS
    assert len(exp_queues) == NUM_AGENTS

    # writer = SummaryWriter(logdir=f'./results/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')
    writer = SummaryWriter()
    logging.basicConfig(level=logging.INFO, 
                        filename='./results/train_log', 
                        filemode='a')

    env = gym.make('CartPole-v1')

    S_DIM = env.observation_space.shape[0]
    A_DIM = env.action_space.n

    a3c = A3C(is_central=True, s_dim=S_DIM, a_dim=A_DIM, actor_lr=ACTOR_LR_RATE, critic_lr=CRITIC_LR_RATE)

    if pre_epoch != -1:
        a3c.actor.load_state_dict(torch.load(os.path.join(MODEL_FOLDER, 'actor_%d.pkl' % pre_epoch)))
        a3c.critic.load_state_dict(torch.load(os.path.join(MODEL_FOLDER, 'critic_%d.pkl' % pre_epoch)))

    for epoch in tqdm(range(EPOCH)):
        # epoch += pre_epoch
        actor_param, critic_param = a3c.actor.state_dict(), a3c.critic.state_dict()
        for i in range(NUM_AGENTS):
            net_param_queues[i].put([actor_param, critic_param])
            # print(i)

        total_reward = []

        for i in range(NUM_AGENTS):
            # print(f'Central agent get batch from agent {i}')
            s_batch, a_batch, r_batch, done = exp_queues[i].get()

            s_batch = torch.from_numpy(np.stack(s_batch)).float()
            a_batch = torch.tensor(a_batch).float()
            r_batch = torch.tensor(r_batch).float()

            a3c.train(s_batch, a_batch, r_batch, done, epoch)

            total_reward.append(torch.sum(r_batch))
            # total_batch_len += len(r_batch)

        a3c.update()
        # print(epoch)

        avg_reward = np.mean(total_reward)

        if (epoch + 1) % MODEL_SAVE_INTERVAL == 0:
            
            # print(epoch)
            torch.save(a3c.actor.state_dict(), os.path.join(MODEL_FOLDER, 'actor_%d.pkl' % (epoch + 1)))
            torch.save(a3c.critic.state_dict(), os.path.join(MODEL_FOLDER, 'critic_%d.pkl' % (epoch + 1)))

            reward_mean = eval_model(epoch + 1)
            logging.info('Epoch %d, train reward: %f, test reward: %f, td loss: %f' % (epoch + 1, avg_reward, reward_mean, a3c.td_loss))

            writer.add_scalar('train/avg_reward', avg_reward, epoch + 1)
            writer.add_scalar('test/avg_reward', reward_mean, epoch + 1)
            writer.add_scalar('train/td_loss', a3c.td_loss, epoch + 1)

            writer.flush()


def agent(pre_epoch, agent_id, net_param_queue, exp_queue):
    env = gym.make('CartPole-v1')

    S_DIM = env.observation_space.shape[0]
    A_DIM = env.action_space.n

    actor = A3C(is_central=False, s_dim=S_DIM, a_dim=A_DIM, actor_lr=ACTOR_LR_RATE, critic_lr=CRITIC_LR_RATE)
    actor_net_params, _ = net_param_queue.get()
    actor.actor.load_state_dict(actor_net_params)

    s_batch = []
    a_batch = []
    r_batch = []

    while True:
        s, _ = env.reset()
        done = False

        # actor_net_params, _ = net_param_queue.get()
        # actor.actor.load_state_dict(actor_net_params)

        while not done:
            a = actor.select_action(torch.from_numpy(s).float().unsqueeze(0))
            s_, r, done, truncated, info = env.step(a)
            done = done or truncated

            s_batch.append(s)
            a_batch.append(a)
            r_batch.append(r)
            s = s_

            if len(s_batch) >= TRAIN_SEQ_LEN or done:
                # print(f'Agent {agent_id} send batch to central agent')
                exp_queue.put([s_batch, a_batch, r_batch, done])
                s_batch = []
                a_batch = []
                r_batch = []

                actor_net_params, _ = net_param_queue.get()
                actor.actor.load_state_dict(actor_net_params)

def main(epoch):
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    net_param_queue = [mp.Queue(1) for _ in range(NUM_AGENTS)]
    exp_queue = [mp.Queue(1) for _ in range(NUM_AGENTS)]

    coordinator = mp.Process(target=central_agent, args=(epoch, net_param_queue, exp_queue))
    coordinator.start()

    agents = [mp.Process(target=agent, args=(epoch, i, net_param_queue[i], exp_queue[i])) for i in range(NUM_AGENTS)]
    for agent_process in agents:
        agent_process.start()

    coordinator.join()
    for agent_process in agents:
        agent_process.join()


if __name__ == '__main__':
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=-1)
    args = parser.parse_args()
    main(args.epoch)
