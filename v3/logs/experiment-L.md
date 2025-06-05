runs\May26_10-17-29_LAPTOP-88AV9U3J - Run with GAE
runs\May26_12-52-08_LAPTOP-88AV9U3J - Run with even more sampled agents. Entropy interval set higher to allow model to train naturally.
runs\May26_15-32-07_LAPTOP-88AV9U3J - Lower JSD threshold to be what is expected.
runs\May26_17-09-24_LAPTOP-88AV9U3J - Set epsilon end to be way highr (from 0.2 to 0.5). Matching how many agents in the population were sampled.
runs\May26_18-43-49_LAPTOP-88AV9U3J - Set sample back to a lower value
runs\May26_21-45-08_LAPTOP-88AV9U3J - Use re-indexing per time step. Reduce entropy target from 0.2 -> 0.05 since we are sampling many agents  - Good Run

runs\May27_09-45-07_LAPTOP-88AV9U3J - Train for more than one step  per experience sample.          - Effective. 
runs\May27_13-32-05_LAPTOP-88AV9U3J - Fix: No initial obs. JSD Thresh to 0.5
runs\May27_15-49-44_LAPTOP-88AV9U3J - JSD THresh back to 1.0. Increase steps per epoch to 16
runs\May27_16-16-30_LAPTOP-88AV9U3J -                                                                   Successefully learnt Nash Eq. Checck checkpoint 1500
    outer_loops = 2_000,
    
    actor_learning_rate= 1e-4,
    critic_learning_rate = 1e-3,
    hypernet_learning_rate = 5e-4,

    hypernet_jsd_threshold = 1.0,
    hypernet_samples = 3000,
    hypernet_jsd_weight = 1.0,
    hypernet_entropy_weight = 0.01, 
    hypernet_diversity_weight= 1.0,

    sampled_agents_proportion = 0.5,
    experience_sampling_steps = 10,
    experience_buffer_size = 10,

    entropy_coeff = 0.2,

    epsilon_period = 200,
    epsilon_end = 0.05,

    entropy_target = 0.5,

    eval_temp = -1.0,
    
    # verbose = False,
    device = parameters.device,
    steps_per_epoch = 16


runs\May27_22-01-38_LAPTOP-88AV9U3J - Run the tests on the hypernetwork environment - model successfully learnt different agent behaviors
runs\May28_09-09-14_LAPTOP-88AV9U3J - same as above but with jsd set to 0.5    - Performs better than with jsd set to 1.0. 


runs\May28_20-18-13_LAPTOP-88AV9U3J - Same as runs\May27_16-16-30_LAPTOP-88AV9U3J  but with jsd_threshold = 2.0 so that diversity is measured across both types. 
runs\May29_02-47-10_LAPTOP-88AV9U3J - Same as above but with no entropy.

runs\May29_09-11-39_LAPTOP-88AV9U3J - Made exploration noise scaled to the logits' stddev. This way noise can be more impactful
    outer_loops = 2_000,
    
    actor_learning_rate= 1e-4,
    critic_learning_rate = 1e-3,
    hypernet_learning_rate = 5e-4,

    hypernet_jsd_threshold = 1.0,
    hypernet_samples = 3000,
    hypernet_jsd_weight = 1.0,
    hypernet_entropy_weight = 0.01, 
    hypernet_diversity_weight= 1.0,

    sampled_agents_proportion = 0.5,
    experience_sampling_steps = 10,
    experience_buffer_size = 10,

    entropy_coeff = 0.2,

    epsilon_period = 200,
    epsilon_end = 0.1,
    noise_scale = 3.0,

    entropy_target = 0.5,

    eval_temp = -1.0,
    
    # verbose = False,
    device = parameters.device,
    steps_per_epoch = 16


    runs\May31_00-16-22_LAPTOP-88AV9U3J - changed how the noise parameter operates to be only on modified logits. 
        outer_loops = 2_000,
    
    actor_learning_rate= 1e-4,
    critic_learning_rate = 1e-3,
    hypernet_learning_rate = 5e-4,

    hypernet_jsd_threshold = 1.0,
    hypernet_samples = 3000,
    hypernet_jsd_weight = 1.0,
    hypernet_entropy_weight = 0.01, 
    hypernet_diversity_weight= 1.0,

    sampled_agents_proportion = 0.5,
    experience_sampling_steps = 10,
    experience_buffer_size = 10,

    entropy_coeff = 0.2,

    epsilon_period = 200,
    epsilon_end = 0.1,
    noise_scale = 1.0,

    entropy_target = 0.5,

    eval_temp = -1.0,
    
    # verbose = False,
    device = parameters.device,
    steps_per_epoch = 16



runs\Jun05_11-07-48_LAPTOP-88AV9U3J - Decoder Run for Test Case 2
runs\Jun05_16-01-15_LAPTOP-88AV9U3J - Decoder Run for Test Case 2 (cont. of previous run above)
training_parameters = TrainingParameters(
    outer_loops = 8_000,
    
    actor_learning_rate= 1e-4,
    critic_learning_rate = 1e-4,
    hypernet_learning_rate = 1e-4,
    decoder_learning_rate= 1e-4,
    filter_learning_rate= 1e-4,

    hypernet_jsd_threshold = 2.0,
    hypernet_samples = 3000,
    hypernet_jsd_weight = 1.0,
    hypernet_entropy_weight = 0.01, 
    hypernet_diversity_weight= 1.0,

    sampled_agents_proportion = 0.2,
    experience_sampling_steps = 10,
    experience_buffer_size = 10,

    entropy_coeff = 0.2,

    epsilon_period = 200,
    epsilon_end = 0.05,

    entropy_target = 0.5,

    eval_temp = -1.0,
    eval_k = env.n_types,
    # verbose = False,
    device = parameters.device,
    steps_per_epoch = 4,

)