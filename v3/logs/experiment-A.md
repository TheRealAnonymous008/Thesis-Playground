runs\May26_12-08-54_LAPTOP-JEP06EM9             Check if throwing out the entire batch per epoch works better.
runs\May26_14-43-42_LAPTOP-JEP06EM9             Same as above but no entropy restarts
runs\May26_14-43-42_LAPTOP-JEP06EM9             Betetr run same as above
runs\May26_19-09-17_LAPTOP-JEP06EM9             Set entropy coeff to 1
runs\May26_21-51-05_LAPTOP-JEP06EM9             Same setup as 21-45-08-L but with epsilon interval set more frequently.
runs\May27_08-04-45_LAPTOP-JEP06EM9             
    experience_sampling_steps = 10,
    experience_buffer_size = 10, equivalent to just two games per epoch.  Otherwise equivalent to 27-06-16-42-L

runs\May27_10-04-16_LAPTOP-JEP06EM9             Same as above but with both changed variables as 5 (so buffer only fits 1 game)

runs\May27_11-51-08_LAPTOP-JEP06EM9             Same as 27-09-45-07-L but with JSD threshold  set to expected 
runs\May27_20-21-39_LAPTOP-JEP06EM9             Sane as 27-16-16-30-L but with JSD threshold set to 0.5
runs\May27_15-26-07_LAPTOP-JEP06EM9             Same as above but with steps set to 5000

runs\May28_20-23-56_LAPTOP-JEP06EM9 -           Sampled agents set to 0.25 instead. Comparable to 28-09-09-14-L
training_parameters = TrainingParameters(
    outer_loops = 2_000,
    
    actor_learning_rate= 1e-4,
    critic_learning_rate = 1e-3,
    hypernet_learning_rate = 5e-4,

    hypernet_jsd_threshold = 1.0,
    hypernet_samples = 3000,
    hypernet_jsd_weight = 1.0,
    hypernet_entropy_weight = 0.01, 
    hypernet_diversity_weight= 1.0,

    sampled_agents_proportion = 0.25,
    experience_sampling_steps = 10,
    experience_buffer_size = 10,

    entropy_coeff = 0.2,

    epsilon_period = 200,
    epsilon_end = 0.05,

    entropy_target = 0.5,

    eval_temp = -1.0,
    eval_k = 10,
    # verbose = False,
    device = parameters.device,
    steps_per_epoch = 16
)
train_model(model, env, training_parameters)

runs\May30_21-45-15_LAPTOP-JEP06EM9 - Same as 31_00-16-22-L but with using uniform loggits