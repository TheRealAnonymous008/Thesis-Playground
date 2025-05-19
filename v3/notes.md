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
