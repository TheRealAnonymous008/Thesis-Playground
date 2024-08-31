# Misc

Need to figure out how to have assemblers take specific orders from the list of orders in the demand
* Assemblers then need to figure out how to signal to effectors that this is the target product.
* Could train a small neural network to value each product in the inventory and select the best
    * Embedder needs to be the first layer since products can have varying dims. 

Need to figure out how to have assemblers make partial products of orders from the current order it is aiming for
* Could tie this to a decision / action  (make a part of the current product or the whole thing?)
* A partial product could just be the original order subject to transformations
    * Crop the product demanded (do only a part of it)
    * Mutate a part (transform it to something else / delete it)


Idea for training the model
* Train RL agents first on a fixed layout of the factory. 
* Then train GAN but with a fixed league of the RL agents. That is, run the simulation on the current epoch of RL agents
* Repeat loop


# (Unimplemented) Ideas for MARL solution
1. Use MAPPO (Multi Agent PPO) but modify it 
2. Use a Learned Reawrd Function as in the work of Miyake (2024) 
3. Use a CTDE scheme