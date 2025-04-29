from dataclasses import dataclass

@dataclass
class ParameterSettings :
    n_agents : int = 100

    d_traits : int =  32
    d_beliefs : int = 32 
    d_het_latent : int = 32

    d_het_weights : int = 64
    d_relation : int = 16
    d_message : int = 8
    
    
    d_obs : int = 8
    d_comm_state : int = 32
    d_action : int = 8

    device : str = "cpu"