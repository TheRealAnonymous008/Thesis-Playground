# Custom Implementation of the IDQN network 

from torch._tensor import Tensor
from .base import *
import numpy.random as random

class IDQN(BaseModel): 
    """
    Implements IDQN 
    """

    def __init__(self, 
                 env : CustomGymEnviornment, 
                 model : ComplexModel,
                 feature_extractor : T_FeatureExtractor,
                 target_net : nn.Module,
                 buffer_size : int = 10000, 
                 batch_size : int = 64, 
                 gamma: float = 0.99, 
                 optimizer : T_Optimizer = optim.Adam,
                 loss_fn : T_Loss = nn.MSELoss(),
                 lr : float = 1e-3,
                 tau: float = 1e-3,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 device : str = "cpu",
                 ):
        """
        Initialize the model 

        :param env: The environment to learn from
        :param policy_net:  The policy network. 
        :param feature_extractor: The feature extractor to apply to each state. Note that it must return a tensor with a batch dimension already defined.
        :param buffer_size: The size of the experience replay buffer
        :param batch_size: Learning batch size
        :param gamma: Discount factor for future rewards.
        :param optimizer: The optimizer class to be used for training (default assumed to be ADAM)
        :papram loss_fn: The loss function used for the optimizer
        :param lr: Learning rate for the optimizer  

        :param target_net: The target network.
        :param tau: Soft update parameter for the target network
        :param epsilon_start: Start value for epsilon decay.
        :param epsilon_end: End value for epsilon decay.
        :param epsilon_decay: How much does epsilon decay. Used for epsilon greedy action sampling
        """
        super().__init__(env = env, 
                         model= model,
                         feature_extractor= feature_extractor,
                         buffer_size = buffer_size, 
                         batch_size= batch_size, 
                         gamma = gamma, 
                         optimizer = optimizer, 
                         loss_fn= loss_fn,
                         lr = lr ,
                         device = device
        )
        self.target_net : torch.nn.Module= target_net
        self.target_net.load_state_dict(self._model._policy_net.state_dict())
        self.target_net.eval()
        self.tau : float = tau 
        self.epsilon : float  = epsilon_start
        self.epsilon_end : float = epsilon_end
        self.epsilon_decay : float = epsilon_decay 

        self.target_net.to(self.device)
    
    def learn(self, total_timesteps: int, optimization_passes : int):
        self._model.eval()
        super().learn(total_timesteps)

        for _ in range (optimization_passes): 
            experiences = self.sample_experiences()
            self.optimize_model(experiences)
            
        self.epsilon = max(self.epsilon_end, self.epsilon_decay * self.epsilon)

    def _select_action(self, agent_id: int, state: dict, deterministic: bool = False) -> Tensor:
        """
        Select an action following an epsilon greedy policy, or greedy if deterministic=True
        """
        if deterministic:
            with torch.no_grad():
                action_values = self._model._policy_net.forward(agent_id, state) + self.get_action_mask(agent_id , self.device)
                return torch.argmax(action_values, dim=1)
        
        sample = random.random()
        if sample > self.epsilon:
            with torch.no_grad():
                action_values = self._model._policy_net.forward(agent_id, state) + self.get_action_mask(agent_id , self.device)
                return torch.argmax(action_values, dim=1)
        else:
            return torch.tensor([self.env.action_space(agent_id).sample()], dtype=torch.long)

    def optimize_model(self, experiences):
        """
        Perform a learning step: update the policy network using a batch of experiences.
        """
        states, actions, rewards, next_states, dones = experiences

        agents = self.env.agents
        average_loss = 0

        self._model.train()

        for i, agent in enumerate(agents) : 
            # Compute Q(s_t, a)
            state_action_values = self._model._policy_net.forward(agent, states)
            action : torch.Tensor = actions[agent]
            state_action_values = state_action_values.gather(1, action).squeeze(1)

            # Compute V(s_{t+1}) using the target network
            with torch.no_grad():
                next_state_values = self.target_net.forward(agent, next_states).max(1)[0]
                expected_state_action_values = (next_state_values * self.gamma * (1 - dones[i])) + rewards[i]

            # Compute loss
            loss = self.loss_fn(state_action_values, expected_state_action_values)

            # Optimize the model
            self.policy_optimizer.zero_grad()
            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()

            loss.backward()
            average_loss += loss.item()

            self.policy_optimizer.step()
            self.encoder_optimizer.step()
            self.decoder_optimizer.step()

        print(f"Average loss {average_loss / len(agents)}")

        # Soft update the target network
        self.soft_update()

    def soft_update(self):
        """
        Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for target_param, local_param in zip(self.target_net.parameters(), self._model._policy_net.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)