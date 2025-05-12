One optimization we can do for the heterogeneous weights is to not actually include it in the network.
Instead, precompute everything homogeneous before applying heterogeneous weights 




# Key Observations
* For the simplest environment, it only learns quickly if the game is symmetric and the nash equilibrium action for both agents is the same action. Otherwise, convergence is slower. 

* The hypernet might actually be able to generate diversity.



# TO Test
* Try modifying the exploration noise to instead be temperature based ( Boltzmann exploration )

* Try modifying the policy objective to instead be based on the unperturbed logits. 

* Figure out a way to make it so agents' rewards don't get cancelled out. 
* Try augmenting the reward to be square (since agents are not cooperative)

* Modify SND to be grouped based on observations (via clustering)

* Use LayerNorm for the hypernet 
* Use GAE. 

* Change activation from LeakyRELU to RELU


# Run Logs

1. No hypernet. No epsilon greedy
2. No hypernet. Epsilon greedy = 300 
3. Enable Hypernet. No Epsilon greedy
4. Enable Hypernet. Epsilon greedy