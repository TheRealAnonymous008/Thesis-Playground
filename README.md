# Design TODOs

- Make the model learn for a specified number of iterations after sampling
- Refactor metric visualization to its own file / functions


- Model agent desires and preferences
- Add action masks
- Add crafting time in the recipe.
- Need to refactor the agent detection as well so that it uses the same logic as that of the resource map.
- Refactor to only call the utility function when the agent executed an action that has potentially changed their current utility (i.e., anything but move and idle).

# (Unimplemented) Ideas for MARL solution
1. Use MAPPO (Multi Agent PPO) but modify it 
2. Use a Learned Reawrd Function as in the work of Miyake (2024) 
3. Use a CTDE scheme