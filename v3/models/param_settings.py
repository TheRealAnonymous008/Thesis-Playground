from dataclasses import dataclass

@dataclass
class ParameterSettings :
    n_agents : int = 100

    d_traits : int =  32
    d_beliefs : int = 32 
    d_het_latent : int = 32

    d_het_weights : int = 64
    d_relation : int = 16
    d_message : int = 8
    
    
    d_obs : int = 8
    d_comm_state : int = 32
    d_action : int = 8

    device : str = "cpu"

    hypernet_scale_factor : float = 50


@dataclass
class TrainingParameters:
    outer_loops: int = 5
    actor_learning_rate: float = 1e-3
    critic_learning_rate : float = 1e-3
    device : str = "cuda"

    gamma: float = 0.99  # Discount factor
    experience_buffer_size : int = 3         # Warning: You shouldn't make this too big because you will have many agents in the env
    actor_performance_weight : float = 1.0
    experience_sampling_steps : int = 100
    grad_clip_norm = 10.0

    # Exploration specific
    entropy_target: float = 0.1  # Target entropy for exploration. Used in the hypernet.
    noise_scale: float = 0.1     # Scale for parameter noise
    epsilon_start: float = 0.8   # Starting probability for epsilon-greedy
    epsilon_end: float = 0.2    # Ending probability for epsilon-greedy
    epsilon_decay: float = 0.99  # Decay rate for epsilon
    epsilon_period: int = 250    # period for cosine scheduling. If 0, doesn't use cosine scheduling
    
    
    # PPO-specific parameters
    clip_epsilon: float = 0.2
    value_loss_coeff: float = 1.0
    entropy_coeff: float = 0.2

    # Hypernet specific parameters
    hypernet_learning_rate : float = 1e-3
    hypernet_entropy_weight : float = 0.1
    hypernet_jsd_threshold: float = 0.5  
    hypernet_jsd_weight : float = 1.0
    hypernet_diversity_weight : float = 0.5
    hypernet_samples_per_batch : float = 1.0         # A constant determining how many to sample

    sampled_agents_proportion : float = 1.0

    filter_learning_rate : float = 1e-3
    decoder_learning_rate : float = 1e-3

    # Control training flow here
    should_train_hypernet : bool = True,
    should_train_actor : bool = True 
    verbose : bool = True

    ppo_epochs: int = 15
    hypernet_steps : int = 15


    # SAC Specific 
    alpha : float = 0.2
    automatic_entropy_tuning : bool = True
    target_entropy : float = 0.01
    tau : float = 0.005


    # Do not change this
    global_steps : int = 0
    epsilon : float = 0 

    # Other stuff
    eval_temp : float = 1.0