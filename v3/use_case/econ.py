import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np
import pandas as pd 
from enum import Enum 
from dataclasses import dataclass
from .network import * 

pd.options.mode.chained_assignment = None  # default='warn'

@dataclass
class EconConfig: 
    n_households : int = 1000
    n_c_firms : int = 100
    n_k_firms : int = 20 
    n_banks : int = 3 
    n_c_product_types : int = 10
    n_k_product_types : int = 10
    n_skills : int = 20

    init_house_money : float = 100.0
    init_firm_capital : float = 1000.0 
    base_wage : float = 10.0


    neighbor_sampling_threshold : int = 5
    relation_dims : int = 3

    @property
    def total_agents(self):
        return self.n_households + self.n_c_firms + self.n_banks + self.n_k_firms

class AgentType(Enum):
    HOUSEHOLD = 0
    C_FIRM = 1
    K_FIRM = 2
    BANK = 3



class EconomyEnv(gym.Env):
    def __init__(self, config : EconConfig):
        super().__init__()
        if config is None:
            raise Exception("Config not passed")
        
        self.config = config

        # Initialize agent states using numpy arrays for scalability
        self._init_state()

        # Define observation and action spaces
        self.action_space = self._create_action_space()
        self.observation_space = self._create_observation_space()

    def _init_state(self):
        """Initialize all agents with starting states. Also initialize the different markets treated as relations"""
        self._init_agents()
        self._init_relations()
        self._initialize_network()

    def _init_agents(self):
        """Initialize all agents in the system"""
        n_agents = 0
        self.agent_indices = {}

        self.households = pd.DataFrame([
            {
                "id": i,
                "money": self.config.init_house_money,
                "wants": np.zeros(self.config.n_c_product_types),
                "inventory": np.zeros(self.config.n_c_product_types),
                "skills" : np.zeros(self.config.n_skills),
            }
            for i in range(self.config.n_households)
        ])
        self.households.set_index("id", inplace= True)
        n_agents = self.config.n_households
        self.agent_indices[AgentType.HOUSEHOLD] =  self.households.index.to_numpy()

        self.c_firms = pd.DataFrame([
            {
                "id" : n_agents + i,
                "product_type" : np.random.choice(self.config.n_c_product_types),
                "quantity": 0,
                "price": 0, 
                "money": self.config.init_firm_capital,
                "capital_inventory" : np.zeros(self.config.n_k_product_types)
            }
            for i in range(self.config.n_c_firms)
        ])
        self.c_firms.set_index("id", inplace= True)
        n_agents += self.config.n_c_firms
        self.agent_indices[AgentType.C_FIRM] = self.c_firms.index.to_numpy()

        self.k_firms = pd.DataFrame([
            {
                "id" : n_agents + i,
                "product_type" : np.random.choice(self.config.n_k_product_types),
                "quantity": 0,
                "price": 0, 
                "money": self.config.init_firm_capital,
            }
            for i in range(self.config.n_k_firms)
        ])
        self.k_firms.set_index("id", inplace= True)
        n_agents += self.config.n_k_firms
        self.agent_indices[AgentType.K_FIRM] = self.k_firms.index.to_numpy()

        self.banks = pd.DataFrame([
            {
                "id" : n_agents + i,
                "deposits": 0, 
                "loans": 0, 
                "interest_rate": 0.05
            }
            for i in range(self.config.n_k_firms)
        ])

        self.banks.set_index("id", inplace= True)
        self.agent_indices[AgentType.BANK] = self.banks.index.to_numpy()


    def _init_relations(self):
        """Initialize the different markets as well as the network topology"""
        c_amt = min(100, self.config.n_households)
        k_amt = min(100, self.config.n_households)
        X = np.random.choice(self.households.index, c_amt + k_amt, replace=False)
        Y = np.random.choice(self.c_firms.index, c_amt)
        Z = np.random.choice(self.k_firms.index, k_amt)

        # Labor Market
        self._labor_market = pd.concat([
            # C firms
            pd.DataFrame([
                {
                    "employer": Y[i],
                    "employee": X[i],
                    "type": AgentType.C_FIRM, 
                    "wage": 1,
                } 
                for i in range(0, len(Y))
            ]), 

            # K Firms
            pd.DataFrame([
                {
                    "employer": Z[i],
                    "employee": X[i + c_amt], 
                    "type": AgentType.K_FIRM, 
                    "wage": 1,
                }
                for i in range(0, len(Z))
            ])
        ])

    def _initialize_network(self):
        """
        Create the relation network.
        """
        # The general communications network 
        self._network : Network = Network(self.config.total_agents, self.config.relation_dims)

        
        m0 = 2  # Initial number of nodes
        m = 2   # Number of edges to attach from each new node

        # Initialize degrees array
        degrees = np.zeros(self.config.total_agents, dtype=int)

        # Initialize the network with m0 nodes connected in a complete graph
        for i in range(m0):
            for j in range(i + 1, m0):
                # Add edges in both directions with random weights
                weight_i_j = np.random.rand(self.config.relation_dims).astype(np.float32)
                self._network.add_connection(i, j, weight_i_j)
                weight_j_i = np.random.rand(self.config.relation_dims).astype(np.float32)
                self._network.add_connection(j, i, weight_j_i)
                degrees[i] += 1
                degrees[j] += 1

        # Add remaining nodes with preferential attachment
        for new_node in range(m0, self.config.total_agents):
            available_nodes = list(range(new_node))  # All existing nodes before new_node
            if not available_nodes:
                continue  # Skip if no available nodes (unlikely as m0 >= 2)

            # Calculate probabilities based on current degrees
            total_degree = degrees[available_nodes].sum()
            if total_degree == 0:
                # If all available nodes have degree 0, choose uniformly
                probs = None
            else:
                probs = degrees[available_nodes] / total_degree

            # Determine how many edges to add (m or as many as possible)
            possible_edges = min(m, len(available_nodes))
            if possible_edges <= 0:
                continue

            # Select m nodes without replacement
            selected = np.random.choice(
                available_nodes,
                size=possible_edges,
                replace=False,
                p=probs
            )

            for s in selected:
                # Add edges in both directions with random weights
                weight_new_s = np.random.rand(self.config.relation_dims).astype(np.float32)
                self._network.add_connection(new_node, s, weight_new_s)
                weight_s_new = np.random.rand(self.config.relation_dims).astype(np.float32)
                self._network.add_connection(s, new_node, weight_s_new)

                # Update degrees
                degrees[new_node] += 1
                degrees[s] += 1

    def _create_action_space(self):
        """Create composite action space for all agent types"""

        # TODO: Add to this
        """
        List of events:
        - Households choose their consumption budget
        - Firms (both C and K type) set their quantity and price.
        - Unemployed households try to find employment  / Employed households evaluate whether they want to stay in the job.
        - Households buy product from C-firms  / C-firms buy capital from K-firms. 
        - Agents communicate with their neighbors to exchange information. 

        Non-Communication Action spaces
        Households: 
        - Establish consumption budget 
        - Make goods transactions
        - Make labor transactions.

        Alternative Idea; the subspaces in the action space should now correspond to one context rather than one agent. This makes it easier
        to do everything in one go.  
        """


        return spaces.Dict({
            # Capital and Goods Market 
            
            "household_budget": spaces.Box(low=0, high=1, shape=(self.config.n_households, )),               # The budget represents the percentage of money allocated for consumption
            "c_firm_price_quantity": spaces.Box(low=0, high=np.inf, shape=(self.config.n_c_firms, 2)),       # [price, quantity] 
            "k_firm_price_quantity": spaces.Box(low=0, high=np.inf, shape=(self.config.n_k_firms, 2)),       # [price, quantity] 
        })

    def _create_observation_space(self):
        """Create composite observation space for all agent types"""

        return spaces.Dict({
            'household': spaces.Dict({
                "money": spaces.Box(low=0, high = np.inf, shape=(self.config.n_households, )),
                "wage": spaces.Box(low = 0, high = np.inf, shape = (self.config.n_households, )),
                "wants" : spaces.Box(low = 0, high = np.inf, shape = (self.config.n_households, self.config.n_c_product_types)),
                "inventory" : spaces.Box(low = 0, high = np.inf, shape = (self.config.n_households, self.config.n_c_product_types)),
                "budget" : spaces.Box(low = 0, high = np.inf, shape = (self.config.n_households, self.config.n_c_product_types)),
            }),

            'c_firms' : spaces.Dict({
                "money": spaces.Box(low=0, high = np.inf, shape=(self.config.n_c_firms, )),
                "quantity" : spaces.Box(low = 0, high = np.inf, shape=(self.config.n_c_firms, )),
                "price" : spaces.Box(low = 0, high = np.inf, shape=(self.config.n_k_firms, )),
                "capital_inventory": spaces.Box(low = 0, high = np.inf, shape = (self.config.n_c_firms, self.config.n_k_product_types)),
            }),

            'k_firms' : spaces.Dict({
                "money": spaces.Box(low=0, high = np.inf, shape=(self.config.n_k_firms, )),
                "quantity" : spaces.Box(low = 0, high = np.inf, shape=(self.config.n_k_firms, )),
                "price" : spaces.Box(low = 0, high = np.inf, shape=(self.config.n_k_firms, )),
            }),  
        })

    def reset(self):
        """Reset the environment to initial state"""
        self._init_state()
        return self._get_observations()

    def step(self, actions):
        """Execute one timestep of the economic simulation"""
        # Process actions

        # Run economic processes
        self._simulate_labor_market()
        self._simulate_banking_operations()

        # Perform the different processes
        self.goods_consumption(actions)


        # Get observations and calculate rewards
        obs = self._get_observations()
        rewards = self._calculate_rewards()
        done = False  # Continuous environment
        info = {}

        return obs, rewards, done, info

    def _simulate_labor_market(self):
        """Simulate labor market operations"""
        # Aggregate wages per employee and update households
        employee_wages = self._labor_market.groupby('employee')['wage'].sum()
        self.households.loc[employee_wages.index, 'money'] += employee_wages

        # Process C_FIRM wage deductions
        c_firm_mask = self._labor_market['type'] == AgentType.C_FIRM
        c_firm_wages = self._labor_market[c_firm_mask].groupby('employer')['wage'].sum()
        self.c_firms.loc[c_firm_wages.index, 'money'] -= c_firm_wages

        # Process K_FIRM wage deductions
        k_firm_mask = self._labor_market['type'] == AgentType.K_FIRM
        k_firm_wages = self._labor_market[k_firm_mask].groupby('employer')['wage'].sum()
        self.k_firms.loc[k_firm_wages.index, 'money'] -= k_firm_wages


    def _simulate_banking_operations(self):
        """Simulate banking system operations"""
        pass 
    
    def goods_consumption(self, actions): 
        """ Processes revolving around budgetting and consuming goods"""
        self.households["budget"] =  actions["household_budget"]

    def make_transaction(self, buyer, seller, quantity):
        """Perform a transaction between a household (buyer) and a c_firm (seller)."""
        # Retrieve seller's product details
        product_type = self.c_firms.loc[seller, 'product_type']
        available_qty = self.c_firms.loc[seller, 'quantity']
        price = self.c_firms.loc[seller, 'price']
        
        # Retrieve buyer's available money
        buyer_money = self.households.loc[buyer, 'money']
        
        # Check if transaction is possible
        if price <= 0 or available_qty <= 0 or buyer_money <= 0:
            return 0  # No transaction possible
        
        # Calculate maximum quantity that can be bought
        max_qty = min(available_qty, quantity, buyer_money / price)
        
        if max_qty <= 0:
            return 0  # No transaction
        
        total_cost = max_qty * price
        
        # Update buyer's money
        self.households.loc[buyer, 'money'] -= total_cost
        
        # Update seller's money and quantity
        self.c_firms.loc[seller, 'money'] += total_cost
        self.c_firms.loc[seller, 'quantity'] -= max_qty
        
        # Update buyer's inventory for the specific product type
        self.households.loc[buyer, 'inventory'][product_type] += max_qty
        
        return max_qty  # Return the actual quantity transacted

    def _calculate_rewards(self):
        """Calculate rewards for all agents"""
        return {
            "household" : self.households["money"].to_numpy() ,
            "c_firms":  self.c_firms["money"].to_numpy(),
            "k_firms": self.k_firms["money"].to_numpy(),
        }

    def _get_observations(self):
        """Compile observations for all agents"""
        employee_wages = self._labor_market.groupby('employee')['wage'].sum()
        wage_observations = employee_wages.reindex(self.households.index, fill_value=0)
        # TODO: This may be completely unnecessary. We amy opt instead to just return the dataframes directly. 
        return {
            "household" : {
                "money": self.households["money"].to_numpy(),
                "wage": wage_observations.to_numpy(),
                "wants": self.households["wants"].to_numpy(),
                "inventory": np.vstack(self.households["inventory"].values),
            }, 

            "c_firms" : {
                "money" : self.c_firms["money"].to_numpy(),
                "quantity" : self.c_firms["quantity"].to_numpy(),
                "price": self.c_firms["price"].to_numpy(),
                "capital_inventory": self.c_firms["capital_inventory"].to_numpy()
            },

            "k_firms" : {
                "money" : self.k_firms["money"].to_numpy(),
                "quantity" : self.k_firms["quantity"].to_numpy(),
                "price": self.k_firms["price"].to_numpy(),
            }

        }
    
    def get_agents(self, agent_type : AgentType):
        arr : np.ndarray = self.agent_indices[agent_type].copy()
        np.random.shuffle(arr)
        return arr 