#few changes to 
#https://github.com/shariqiqbal2810/maddpg-pytorch/blob/master/utils/networks.py

import torch
import torch.nn as nn
import torch.nn.functional as F



class Actor(nn.Module):
    """
    Neural (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=nn.ELU,
                 constrain_out=False, norm_in=True, discrete_action=False):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(Actor, self).__init__()

        if norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(input_dim)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.nonlin = nonlin
        if constrain_out and not discrete_action:
            # initialize small to prevent saturation
            self.fc3.weight.data.uniform_(-3e-3, 3e-3)
            self.out_fn = F.tanh
        else:  # logits for discrete action (will softmax later)
            self.out_fn = lambda x: x

    def forward(self, state):
        """
        Inputs:
            state (PyTorch Matrix): Batch of observations
            action (PyTorch Matrix): Batch of actions
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        elu = nn.ELU()
        input_d = self.fc1(self.in_fn(state))
        h1 = elu(input_d)
        input_t = self.fc2(h1)
        h2 = elu(input_t)
        final = self.fc3(h2)
        out = elu(final)
        return out

class Critic(nn.Module):
    """
    Neural (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, action_size, hidden_dim=64, nonlin=nn.ELU,
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
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim+action_size, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.nonlin = nonlin
        if constrain_out and not discrete_action:
            # initialize small to prevent saturation
            self.fc3.weight.data.uniform_(-3e-3, 3e-3)
            self.out_fn = F.tanh
        else:  # logits for discrete action (will softmax later)
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
        h1 = elu(input_d)
        x_join = torch.cat((h1, actions), dim=1)

        input_t = self.fc2(x_join)
        h2 = elu(input_t)
        final = self.fc3(h2)
        out = elu(final)
        return out