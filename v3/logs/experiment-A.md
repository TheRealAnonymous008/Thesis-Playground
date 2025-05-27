runs\May26_12-08-54_LAPTOP-JEP06EM9             Check if throwing out the entire batch per epoch works better.
runs\May26_14-43-42_LAPTOP-JEP06EM9             Same as above but no entropy restarts
runs\May26_14-43-42_LAPTOP-JEP06EM9             Betetr run same as above
runs\May26_19-09-17_LAPTOP-JEP06EM9             Set entropy coeff to 1
runs\May26_21-51-05_LAPTOP-JEP06EM9             Same setup as 21-45-08-L but with epsilon interval set more frequently.
runs\May27_08-04-45_LAPTOP-JEP06EM9             
    experience_sampling_steps = 10,
    experience_buffer_size = 10, equivalent to just two games per epoch.  Otherwise equivalent to 27-06-16-42-L

runs\May27_10-04-16_LAPTOP-JEP06EM9             Same as above but with both changed variables as 5 (so buffer only fits 1 game)