#https://github.com/shariqiqbal2810/maddpg-pytorch/blob/master/utils/agents.py
import torch
from torch import Tensor
from torch.autograd import Variable
from torch.optim import Adam
from network import Neural
from utils import hard_update, gumbel_softmax, onehot_from_logits, OUNoise

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DDPGAgent(object):
    """
    General class for DDPG agents (policy, critic, target policy, target
    critic, exploration noise)
    """
    def __init__(self, num_in_pol, num_out_pol, num_in_critic, hidden_dim_actor=120,
    hidden_dim_critic=64,lr_actor=0.01,lr_critic=0.01, discrete_action=False):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
            num_in_critic (int): number of dimensions for critic input
        """
        self.policy = Neural(num_in_pol, num_out_pol,
                                 hidden_dim=hidden_dim_actor,
                                 constrain_out=True,
                                 discrete_action=discrete_action)
        self.critic = Neural(num_in_critic, 1,
                                 hidden_dim=hidden_dim_critic,
                                 constrain_out=False)
        self.target_policy = Neural(num_in_pol, num_out_pol,
                                        hidden_dim=hidden_dim_actor,
                                        constrain_out=True,
                                        discrete_action=discrete_action)
        self.target_critic = Neural(num_in_critic, 1,
                                        hidden_dim=hidden_dim_critic,
                                        constrain_out=False)
        hard_update(self.target_policy, self.policy)
        hard_update(self.target_critic, self.critic)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr_actor)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_critic)
        
        self.exploration = OUNoise(num_out_pol)
        
        self.discrete_action = discrete_action

    def reset_noise(self):
        if not self.discrete_action:
            self.exploration.reset()

    def scale_noise(self, scale):
        if self.discrete_action:
            self.exploration = scale
        else:
            self.exploration.scale = scale

    def step(self, obs, explore=True):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs : Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        state = Variable(torch.Tensor(obs[None, ...]),requires_grad=False)
        
        self.policy.eval()
        with torch.no_grad():
            action = self.policy(state)
        self.policy.train()
        # continuous action
        if explore:
            action += Variable(Tensor(self.exploration.noise()),requires_grad=False)
            action = action.clamp(-1, 1)
        return action

    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'critic': self.critic.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'target_critic': self.target_critic.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict()}

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.critic.load_state_dict(params['critic'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.target_critic.load_state_dict(params['target_critic'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        self.critic_optimizer.load_state_dict(params['critic_optimizer'])