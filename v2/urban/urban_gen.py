from __future__ import annotations
from dataclasses import dataclass, field
from math import dist

from noise import snoise2
from utils.line import *
from core.env_params import MAX_VISIBILITY
from core.terrain_map import *
from typing import Dict, Tuple, List
import heapq

 
RoadNode = Tuple[int, int] 

@dataclass
class RoadSegment:
    start: RoadNode  # Coordinates of the start point
    end: RoadNode    # Coordinates of the end point

@dataclass
class RoadNetwork:
    # The road network is represented as an adjacency list of nodes (intersections/endpoints)
    # The keys are coordinates (nodes) and the values are lists of adjacent road segments
    graph: Dict[RoadNode, List[RoadNode]] = field(default_factory=dict)
    
    def add_road_segment(self, segment: RoadSegment):
        """Adds a road segment to the network, connecting its start and end nodes."""
        if segment.start not in self.graph:
            self.graph[segment.start] = []
        if segment.end not in self.graph:
            self.graph[segment.end] = []
        
        # Add the segment to both the start and end nodes (since roads are bidirectional)
        self.graph[segment.start].append(segment.end)
        self.graph[segment.end].append(segment.start)
    
    def get_neighbors(self, node: RoadNode) -> List[RoadNode]:
        """Returns the list of road nodes connected to a given node (intersection)."""
        return self.graph.get(node, [])


class UrbanTerrainMapGenerator(TerrainMapGenerator):
    """
    Derived class for generating a realistic urban area terrain map.
    """
    def __init__(self, 
                    base_height_range : tuple[int, int] = (-10, 10),
                    building_height_range : tuple[int, int] = (50, 100),
                    padding=MAX_VISIBILITY
                ):
        self.base_height_range =  base_height_range
        self.building_height_range = building_height_range

        min_height = base_height_range[0] 
        max_height = base_height_range[1] + building_height_range[1]
        super().__init__(min_height, max_height, padding)


    def generate(self, dims: tuple[int, int]) -> tuple[TerrainMap, tuple[int, int], tuple[int, int]]:
        """
        Generate a realistic urban terrain map with roads and buildings.
        """

        # Generate the population density 
        population_density = self.generate_population_density(dims)

        # Generate a road network
        road_network = self.generate_road_network(dims)
        # Discretize with the height map
        height_map = np.full((dims[0], dims[1]), (self.min_height + self.max_height) / 2, dtype=np.float32)
        for (start ,v) in road_network.graph.items():
            for end in v: 
                bresenham_line(height_map, start, end)

        # Create the TerrainMap object
        terrain_map = TerrainMap(height_map=height_map, padding=self.padding, density_map = population_density)
        return terrain_map
    

    def generate_population_density(self, dims: tuple[int, int], scale: float = 100.0, octaves: int = 6, persistence: float = 0.5, lacunarity: float = 2.0) -> np.ndarray:
        """
        Generate a population density map using Perlin noise.
        """
        density_map = np.zeros(dims)
        for i in range(dims[0]):
            for j in range(dims[1]):
                # Generate Perlin noise value at (i, j)
                noise_value = snoise2(i / scale, j / scale, octaves=octaves, persistence=persistence, lacunarity=lacunarity)
                
                # Normalize noise_value to range 0.0 to 1.0
                noise_value = (noise_value + 1) / 2.0
                density_map[i, j] = noise_value
        
        return density_map
    
    def generate_road_network(self, dims: tuple[int, int]) -> RoadNetwork:
        """
        Generate a road network
        """
        road_network = RoadNetwork()

        # Define parameters for road generation, e.g., initial road length, angle, etc.
        max_iterations = 100  # Limit to prevent infinite growth
        max_distance_to_merge = 2  # Distance threshold to merge close nodes

        # Step 1: Generate initial roads using an L-system or some growth rule
        start_point = (dims[0] // 2, dims[1] // 2)  # Start road from center
        self.grow_roads(dims, road_network, start_point, max_iterations)

        # Step 2: Mark intersections as nodes
        self.mark_intersections_as_nodes(road_network)

        # Step 3: Merge close nodes
        self.merge_close_nodes(road_network, max_distance_to_merge)

        # Return the road network
        return road_network

    def grow_roads(self, dims : tuple[int, int], road_network: RoadNetwork, start_point: RoadNode, max_iterations: int):
        """Grow roads from a starting point with a simple L-system-like process."""
        directions = [
            (0, 1), 
            (1, 0), 
            (0, -1), 
            (-1, 0),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1)
        ] 
        vertex_list : list[RoadNode] = []
        vertex_list.append(start_point)
        
        for _ in range(max_iterations):
            P0 = vertex_list[np.random.choice(len(vertex_list))]
            direction = directions[np.random.choice(len(directions))]
            length = np.random.randint(1, 6) * 5

            x = P0[0] + direction[0] * length
            y = P0[1] + direction[1] * length
            
            x = min(max(x, 0), dims[0] - 1)
            y = min(max(y, 0), dims[1] - 1)
            P1 = (x, y)

            vertex_list.append(P1)
            road_segment = RoadSegment(start=P0, end = P1)
            road_network.add_road_segment(road_segment)

    def mark_intersections_as_nodes(self, road_network: RoadNetwork):
        """Mark intersections of roads as nodes."""
        for node, segments in road_network.graph.items():
            if len(segments) > 2:  # A node with more than 2 connected segments is an intersection
                # We could apply some logic to mark it, but for now, we assume the graph tracks this naturally
                pass

    def merge_close_nodes(self, road_network: RoadNetwork, max_distance_to_merge: int):
        """Merge nodes that are close to each other."""
        all_nodes = list(road_network.graph.keys())

        for i in range(len(all_nodes)):
            for j in range(i + 1, len(all_nodes)):
                node_a = all_nodes[i]
                node_b = all_nodes[j]

                if dist(node_a, node_b) < max_distance_to_merge:
                    # Merge node_b into node_a (remove node_b and redirect all roads to node_a)
                    neighbors = road_network.graph.pop(node_b, [])

                    for neighbor in neighbors:
                        # Redirect the segment to node_a
                        new_segment = RoadSegment(node_a, neighbor)
                        road_network.add_road_segment(new_segment)
    