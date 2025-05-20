One optimization we can do for the heterogeneous weights is to not actually include it in the network.
Instead, precompute everything homogeneous before applying heterogeneous weights 




# Key Observations
* For the simplest environment, it only learns quickly if the game is symmetric and the nash equilibrium action for both agents is the same action. Otherwise, convergence is slower. 

* The hypernet might actually be able to generate diversity.

* Do not use dropout as it lowers performance.
* Set d_belief to be > 1 (8 seems to work fine)
* Target entropy for the hypernet is important since we don't want it to be too low that agents become homogeneous.

# TO Test

* Use GAE. 
* GNN Training


* Change activation from LeakyRELU to RELU


# Run Logs



Good Params
training_parameters = TrainingParameters(
    outer_loops = 1_000,
    
    actor_learning_rate= 1e-4,
    critic_learning_rate = 1e-3,
    hypernet_learning_rate = 5e-4,

    hypernet_jsd_threshold = 2.0,
    hypernet_samples = 3000,
    hypernet_jsd_weight = 1.0,
    hypernet_entropy_weight = 1.0, 
    hypernet_diversity_weight= 1.0,

    sampled_agents_proportion = 0.1,
    experience_sampling_steps = 10,
    experience_buffer_size = 100,

    entropy_coeff = 0.2,

    epsilon_period = 1000,
    epsilon_end = 0.01,

    entropy_target = 0.5,

    eval_temp = -1.0,
    
    # verbose = False,

    should_train_gnn= False
)