One optimization we can do for the heterogeneous weights is to not actually include it in the network.
Instead, precompute everything homogeneous before applying heterogeneous weights 




# Key Observations
* For the simplest environment, it only learns quickly if the game is symmetric and the nash equilibrium action for both agents is the same action. Otherwise, convergence is slower. 

* The hypernet might actually be able to generate diversity.



# TO Test
* Modify SND to be grouped based on observations (via clustering)

* Use LayerNorm for the hypernet 
* Use GAE. 

* Use a scheduler for the exploration noise.
* Change activation from LeakyRELU to RELU