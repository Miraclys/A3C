import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import torch.optim as optim

GAMMA = 0.99
EPOCH = 100000

HIDDEN_DIM = 128
ACTOR_KERNEL_SIZE = 4

def get_entropy_weight(epoch):
    if epoch >= EPOCH:
        return 0.1
    else:
        return 1 - 0.9 * epoch / EPOCH

class ActorNet(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(ActorNet, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim

        self.fc1 = nn.Linear(s_dim, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.out_layer = nn.Linear(HIDDEN_DIM, a_dim)

    def forward(self, state):
        out_layer = F.relu(self.fc1(state))
        out_layer = F.relu(self.fc2(out_layer))
        logits = self.out_layer(out_layer)
        return logits
    
class CriticNet(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(CriticNet, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim

        self.fc1 = nn.Linear(s_dim, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.out_layer = nn.Linear(HIDDEN_DIM, 1)

    def forward(self, state):
        out_layer = F.relu(self.fc1(state))
        out_layer = F.relu(self.fc2(out_layer))
        value = self.out_layer(out_layer)
        return value

if __name__ == '__main__':
    print((3 + 2 * ((S_LEN - ACTOR_KERNEL_SIZE) / 1 + 1) + \
           ((A_DIM - ACTOR_KERNEL_SIZE) / 1 + 1)) * HIDDEN_DIM)