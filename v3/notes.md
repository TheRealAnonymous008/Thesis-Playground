One optimization we can do for the heterogeneous weights is to not actually include it in the network.
Instead, precompute everything homogeneous before applying heterogeneous weights 




# Key Observations
* For the simplest environment, it only learns quickly if the game is symmetric and the nash equilibrium action for both agents is the same action. Otherwise, convergence is slower. 

* The hypernet might actually be able to generate diversity.



# TODOs
* Hypernet should copy PPO and have steps for optimizing towards target
* Use GAE. 
* Change activation from LeakyRELU to RELU
* Change the SND metric and normalize it by dividing by the number of agent pairs used. 