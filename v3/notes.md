One optimization we can do for the heterogeneous weights is to not actually include it in the network.
Instead, precompute everything homogeneous before applying heterogeneous weights 

For consumption budgeting, instead of having agents learn budgeting strategies (arguably out of scope), just limit iit to greedy behavior or select from a pool of strategies instead. (Prio on greedy budgeting for controllability) 




How do we handle transactions and selecting which neighbor to interact with? 
Idea #1: Feed each agent its neighbor (one at a time) and call the relevant function. 
Downside: What if there are too many neighbors? 

Idea #2: Same as idea #1 but impose a cutoff (i.e., analogous to context window in transformers). Choose up to k neighbors to consider.

Go with Idea 2







