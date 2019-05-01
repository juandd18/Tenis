#few changes to 
#https://github.com/shariqiqbal2810/maddpg-pytorch/blob/master/utils/networks.py

import torch
import torch.nn as nn
import torch.nn.functional as F



class Actor(nn.Module):
    """
    Neural (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, hidden_dim=128, nonlin=nn.ELU,
                  norm_in=True, discrete_action=False):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(Actor, self).__init__()

        self.bn0 = nn.BatchNorm1d(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)

        self.reset_parameters()

    def reset_parameters(self):
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        

    def forward(self, state):
        """
        Inputs:
            state (PyTorch Matrix): Batch of observations
            action (PyTorch Matrix): Batch of actions
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        elu = nn.ELU()
        x = self.bn0(state)
        x = elu(self.bn1(self.fc1(x)))
        #x = elu(self.fc1(x))
        x =elu(self.fc2(x))
        return torch.tanh(self.fc3(x))

class Critic(nn.Module):
    """
    Neural (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, action_size, hidden_dim=128, nonlin=nn.ELU,
                 constrain_out=False, norm_in=True, discrete_action=False):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(Critic, self).__init__()

        if norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim+action_size, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.nonlin = nonlin

        # logits for discrete action (will softmax later)
        self.out_fn = lambda x: x

    def forward(self, state,actions):
        """
        Inputs:
            state (PyTorch Matrix): Batch of observations
            action (PyTorch Matrix): Batch of actions
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        elu = nn.ELU()
        input_d = self.fc1(self.in_fn(state))
        #input_d = self.fc1(state)
        h1 = elu(input_d)
        x_join = torch.cat((h1, actions), dim=1)
        
        input_t = self.fc2(x_join)
        h2 = elu(input_t)
        final = self.fc3(h2)
        out = self.out_fn(final)
        return out