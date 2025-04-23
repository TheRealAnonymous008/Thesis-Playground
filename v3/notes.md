One optimization we can do for the heterogeneous weights is to not actually include it in the network.
Instead, precompute everything homogeneous before applying heterogeneous weights 

For consumption budgeting, instead of having agents learn budgeting strategies (arguably out of scope), just limit iit to greedy behavior or select from a pool of strategies instead. (Prio on greedy budgeting for controllability) 




For firms, set both stock and quantity. For normalization purposes, make the action x instead be of range 0 to 1 so that the quantity sold is 

qty_sold = x * stock

Any unsold stock gets put back in stock. 




Need to define the production equation for each product. For now we may assume that it is determined purely by labor. 


Optimizations: 
- Instead of using dataframe, just use regular numpy arrays and indexing. 
- Make a simplifying assumption that wages are fixed per employee regardless of skill. 




# Key Observations
* For the simplest environment, it only learns quickly if the game is symmetric and the nash equilibrium action for both agents is the same action. Otherwise, convergence is slower. 

* The hypernet might actually be able to generate diversity.


