One optimization we can do for the heterogeneous weights is to not actually include it in the network.
Instead, precompute everything homogeneous before applying heterogeneous weights 




# Key Observations
* For the simplest environment, it only learns quickly if the game is symmetric and the nash equilibrium action for both agents is the same action. Otherwise, convergence is slower. 

* The hypernet might actually be able to generate diversity.

* Do not use dropout as it lowers performance.

# TO Test

* Use GAE. 

* Change activation from LeakyRELU to RELU


# Run Logs
1. Standard Control test
2. Set     hypernet_jsd_threshold = 0.5,
3. Set      sampling steps = 1, and experience buffer size = 25

