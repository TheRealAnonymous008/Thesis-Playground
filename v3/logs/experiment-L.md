runs\May26_10-17-29_LAPTOP-88AV9U3J - Run with GAE
runs\May26_12-52-08_LAPTOP-88AV9U3J - Run with even more sampled agents. Entropy interval set higher to allow model to train naturally.
runs\May26_15-32-07_LAPTOP-88AV9U3J - Lower JSD threshold to be what is expected.
runs\May26_17-09-24_LAPTOP-88AV9U3J - Set epsilon end to be way highr (from 0.2 to 0.5). Matching how many agents in the population were sampled.
runs\May26_18-43-49_LAPTOP-88AV9U3J - Set sample back to a lower value
runs\May26_21-45-08_LAPTOP-88AV9U3J - Use re-indexing per time step. Reduce entropy target from 0.2 -> 0.05 since we are sampling many agents  - Good Run

runs\May27_09-45-07_LAPTOP-88AV9U3J - Train for more than one step  per experience sample.          - Effective. 
runs\May27_13-32-05_LAPTOP-88AV9U3J - Fix: No initial obs.