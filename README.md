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


# Paper stuff
CH. 1 - Intro
    -> Background 
    -> Objectives 
    -> Scope / Limitations
CH. 2 - RRL
CH. 3 - Theoretical Framework (What's the theoretical background of this)
    -> MARL +  Theoretical Stuff needed
CH. 4 - Methodology (What will we do) 
    -> How was the stuff in Ch. 3 implemeneted.
    -> What are the different steps for answering research question
    -> Evaluation / Experimentation pipeline
Ch. 5 - Description of Model  
    -> Architecture
    -> Specific to what I'm doing 

====
Ch. 6 - Results and Discussion
Ch. 7 - Conclusion

APPENDICES (if needed) 
