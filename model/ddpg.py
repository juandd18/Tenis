#https://github.com/shariqiqbal2810/maddpg-pytorch/blob/master/utils/agents.py
import torch
from torch import Tensor
from torch.autograd import Variable
from torch.optim import Adam
from network import Actor,Critic
from utils import soft_update, hard_update, gumbel_softmax, onehot_from_logits, OUNoise,ReplayBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MSELoss = torch.nn.MSELoss()

class DDPGAgent(object):
    """
    General class for DDPG agents (policy, critic, target policy, target
    critic, exploration noise)
    """
    def __init__(self, num_in_pol, num_out_pol, num_in_critic, hidden_dim_actor=120,
    hidden_dim_critic=64,lr_actor=0.01,lr_critic=0.01,batch_size=64,
    max_episode_len=100,tau=0.02,gamma = 0.78,agent_name='one', discrete_action=False):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
            num_in_critic (int): number of dimensions for critic input
        """
        self.policy = Actor(num_in_pol, num_out_pol,
                                 hidden_dim=hidden_dim_actor,
                                 discrete_action=discrete_action)
        self.critic = Critic(num_in_critic, 1,num_out_pol,
                                 hidden_dim=hidden_dim_critic)
        self.target_policy = Actor(num_in_pol, num_out_pol,
                                        hidden_dim=hidden_dim_actor,
                                        discrete_action=discrete_action)
        self.target_critic = Critic(num_in_critic, 1,num_out_pol,
                                        hidden_dim=hidden_dim_critic)
        hard_update(self.target_policy, self.policy)
        hard_update(self.target_critic, self.critic)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr_actor)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_critic)
        
        self.policy = self.policy.float()
        self.critic = self.critic.float()
        self.target_policy = self.target_policy.float()
        self.target_critic = self.target_critic.float()

        self.agent_name = agent_name
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(1e6)
        self.max_replay_buffer_len = batch_size * max_episode_len
        self.replay_sample_index = None
        self.niter = 0

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

    def act(self, obs, explore=True):
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

    def step(self, state, action, reward, next_state, done,t_step):
        
        #TODO fix limit to learn
        # Learn, if enough samples are available in memory
        if len(self.replay_buffer) < self.max_replay_buffer_len:
            
            self.replay_sample_index = self.replay_buffer.make_index(self.batch_size)

            # collect replay samples
            index = self.replay_sample_index
            obs, acs, rews, next_obs, dones = self.replay_buffer.sample_index(index)

            self.update(obs, acs, rews, next_obs, dones,t_step)
        else:
            # Save experience / reward
            self.replay_buffer.add(state, action, reward, next_state, done)


    def update(self, obs, acs, rews, next_obs, dones ,t_step, logger=None):

        #TODO CHECK if the code below improve performance 
        #if not t_step % 100 == 0:  # only update every 100 steps
        #    return
        
        obs = Variable(torch.from_numpy(obs)).float()
        next_obs = Variable(torch.from_numpy(next_obs)).float()
        rews = Variable(torch.from_numpy(rews)).float()
        acs = Variable(torch.from_numpy(acs)).float()
        acs = acs.view(-1, 2)
        
        # --------- update critic ------------ #        
        self.critic_optimizer.zero_grad()
        
        if self.discrete_action: # one-hot encode action
            all_trgt_acs = onehot_from_logits(self.target_policy(next_obs))  
        else:
            all_trgt_acs = self.target_policy(next_obs) 
     
        target_value = (rews + self.gamma *
                        self.target_critic(next_obs,all_trgt_acs) *
                        torch.from_numpy((1 - dones)).float() )
       
        actual_value = self.critic(obs,acs)
        vf_loss = MSELoss(actual_value, target_value.detach())

        # Minimize the loss
        vf_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

        # --------- update actor --------------- #
        self.policy_optimizer.zero_grad()

        if self.discrete_action:
            curr_pol_out = self.policy(obs)
            curr_pol_vf_in = gumbel_softmax(curr_pol_out, hard=True)
        else:
            curr_pol_out = self.policy(obs)
            curr_pol_vf_in = curr_pol_out


        pol_loss = -self.critic(obs,curr_pol_vf_in).mean()
        pol_loss += (curr_pol_out**2).mean() * 1e-3
        pol_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.policy_optimizer.step()

        self.update_all_targets()

        if logger is not None:
            logger.add_scalars('agent%i/losses' % self.agent_name,
                               {'vf_loss': vf_loss,
                                'pol_loss': pol_loss},
                               self.niter)

    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        
        soft_update(self.target_critic, self.critic, self.tau)
        soft_update(self.target_policy, self.policy, self.tau)
        self.niter += 1

   
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