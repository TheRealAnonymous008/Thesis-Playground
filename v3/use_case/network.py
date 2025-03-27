import numpy as np

class Network:
    def __init__(self, N, edge_dim=4):
        self.N = N
        self.edge_dim = edge_dim
        
        # Store neighbors as sorted arrays for binary search efficiency
        self.neighbors = np.empty(N, dtype=object)
        
        # Store edge weights as 2D arrays (neighbor_count × edge_dim)
        self.weights = np.empty(N, dtype=object)
        
        # Initialize empty arrays with proper dimensions
        for i in range(N):
            self.neighbors[i] = np.array([], dtype=np.int32)
            self.weights[i] = np.empty((0, edge_dim), dtype=np.float32)

    def add_connection(self, source, target, weight_vector):
        """Add a directed edge with vector weight"""
        if source == target:
            return
            
        # Convert weight to numpy array if needed
        if not isinstance(weight_vector, np.ndarray):
            weight_vector = np.array(weight_vector, dtype=np.float32)
            
        # Check weight dimensions
        if weight_vector.shape != (self.edge_dim,):
            raise ValueError(f"Weight vector must have shape ({self.edge_dim},)")
            
        # Append new connection
        self.neighbors[source] = np.append(self.neighbors[source], target)
        self.weights[source] = np.vstack([self.weights[source], weight_vector])
        
        # Maintain sorted order for binary search
        sort_order = np.argsort(self.neighbors[source])
        self.neighbors[source] = self.neighbors[source][sort_order]
        self.weights[source] = self.weights[source][sort_order]

    def remove_connection(self, source, target):
        """Remove a directed edge"""
        idx = np.searchsorted(self.neighbors[source], target)
        if idx < len(self.neighbors[source]) and self.neighbors[source][idx] == target:
            self.neighbors[source] = np.delete(self.neighbors[source], idx)
            self.weights[source] = np.delete(self.weights[source], idx, axis=0)

    def get_edge_weight(self, source, target):
        """Retrieve vector weight for a specific edge"""
        idx = np.searchsorted(self.neighbors[source], target)
        if idx < len(self.neighbors[source]) and self.neighbors[source][idx] == target:
            return self.weights[source][idx]
        return None

    def batch_add_connections(self, sources, targets, weight_vectors):
        """Add multiple edges simultaneously"""
        for src, tgt, wv in zip(sources, targets, weight_vectors):
            self.add_connection(src, tgt, wv)

    def get_adjacency_tensor(self):
        """Get full adjacency tensor (N × N × edge_dim)"""
        adj_tensor = np.zeros((self.N, self.N, self.edge_dim), dtype=np.float32)
        for i in range(self.N):
            if len(self.neighbors[i]) > 0:
                adj_tensor[i, self.neighbors[i]] = self.weights[i]
        return adj_tensor