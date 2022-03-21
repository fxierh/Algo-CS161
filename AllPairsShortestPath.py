import time
import math
import sys
import numpy as np

from multiprocessing import Pool, cpu_count
from functools import partial
from typing import List
from DijkstraShortestPath import Heap, Graph, Node


# Bellman-Ford algorithm for the single source shortest path problem. Time complexity = O(mn), space complexity = O(n).
# Return shortest distances if no negative cycle exists else return None.
def bellman_ford(src_vertex: int, adj_list: List[list]) -> None or list:
    n_nodes = len(adj_list)
    prev_row = [math.inf] * n_nodes  # A[i - 1, v] for all vertices v. Initially i = 1.
    prev_row[src_vertex] = 0
    for _ in range(1, n_nodes + 1, 1):
        row = [None] * n_nodes
        for node in range(n_nodes):
            temp = [prev_row[tail_vertex] + edge_cost for tail_vertex, edge_cost in adj_list[node]] \
                   + [prev_row[node]]
            row[node] = min(temp)
        if row == prev_row:  # Early stopping
            return row
        prev_row = row
    return None


# Vectorized Floyd-Warshall algorithm for the APSP problem. Time complexity = O(n^3), space complexity = O(n^2).
# Return the shortest distance between each pair of vertices if no negative cycle exists else return None.
def floyd_warshall(initial_result_plane: np.ndarray) -> np.ndarray or None:
    # Initialization
    current_plane = initial_result_plane
    n_nodes = current_plane.shape[0]

    # Vectorized main loop
    for k in range(0, n_nodes, 1):
        current_plane = np.minimum(np.add.outer(current_plane[:, k], current_plane[k, :]), current_plane)

    # Check if negative cycle presents
    for node in range(n_nodes):
        if current_plane[node, node] < 0:
            return None

    return current_plane


class Node_2(Node):
    def __init__(self, label: int, weight: int or None):
        super().__init__(label)
        self.weight = weight


class Graph_2(Graph):
    def dijkstra(self, src_vertex: int) -> List:
        # Preprocessing
        # Consider the first node to be processed in the beginning
        self.nodes[src_vertex].key = 0
        self.nodes[src_vertex].shortest_distance = 0
        self.nodes[src_vertex].processed = True
        self.processed_nodes.append(self.nodes[src_vertex])
        # Initialize key of each neighbor of the source node
        for neighbor in self.processed_nodes[0].neighbors.keys():
            neighbor.key = self.processed_nodes[0].neighbors[neighbor]
        # Initialize heap with node array
        heap = Heap(arr=[node for node in self.nodes if node.label != src_vertex])
        heap.heapify()

        # Dijkstra
        while len(self.processed_nodes) < len(self.nodes):
            new_node = heap.pop()
            new_node.shortest_distance = new_node.key
            new_node.processed = True
            self.processed_nodes.append(new_node)
            for neighbor in new_node.neighbors.keys():
                # If neighbor of new node is in heap
                if neighbor.heap_index is not None:
                    # Delete neighbor from heap
                    heap.pop(index=neighbor.heap_index)
                    # Recompute neighbor's key
                    neighbor.key = min(neighbor.key, new_node.shortest_distance + new_node.neighbors[neighbor])
                    # Re-insert neighbor into heap
                    heap.insert(neighbor)

        # Compute offset
        pu = self.nodes[src_vertex].weight
        return [node.shortest_distance - pu + node.weight for node in self.nodes]


# Johnson's algo for the APSP problem. Basically 1*Bellman-Ford + n*Dijkstra. Time complexity = O(m*n*logn)
# Notice that the Dijkstra code is not optimized for speed nor multi-processed.
def johnson(adj_list: List[list], all_edges: List[tuple]) -> list:
    # Create new adj_list with vertex S (vertex index n_nodes, has an edge of length 0 toward every other vertex) added.
    n_nodes = len(adj_list)
    for node in range(n_nodes):
        adj_list[node].append((n_nodes, 0))
    adj_list.append([])

    # Run Bellman-Ford to get node weights
    # (the last element corresponds to the weight of the newly added node S and is not useful)
    node_weights = bellman_ford(src_vertex=n_nodes, adj_list=adj_list)

    # Initialize Dijkstra with original graph (without vertex S).
    graph_2 = Graph_2(n_nodes=n_nodes)
    node_array = []
    for node in range(n_nodes):
        node_array.append(Node_2(label=node, weight=node_weights[node]))
    # Edd all edges with edge cost modified
    for tail_node, head_node, orig_edge_cost in all_edges:
        new_edge_cost = orig_edge_cost + node_array[tail_node].weight - node_array[head_node].weight
        assert new_edge_cost >= 0
        node_array[tail_node].neighbors[node_array[head_node]] = new_edge_cost
    graph_2.nodes = node_array

    # Run Dijkstra n times
    res: List[int or list] = [0]*n_nodes
    for src_node in range(n_nodes):
        res[src_node] = graph_2.dijkstra(src_vertex=src_node)
    return res


if __name__ == '__main__':
    graph_id = 'large'
    assert graph_id in ['1', '2', '3', 'large']
    adjacency_list = []
    edges = []  # All edges
    with open("AssignmentData/Data_APSP_" + graph_id + ".txt", 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                nb_nodes, nb_edges = line.strip().split(' ')
                nb_nodes, nb_edges = int(nb_nodes), int(nb_edges)
                adjacency_list = [[] for _ in range(nb_nodes)]
            else:
                tail, head, length = line.strip().split(' ')
                tail, head, length = int(tail), int(head), int(length)
                adjacency_list[head - 1].append((tail - 1, length))
                edges.append((tail - 1, head - 1, length))
    print(f"Graph {graph_id} starts:")
    result = bellman_ford(src_vertex=0, adj_list=adjacency_list)
    if result is None:
        sys.exit("Negative cycle present, exit program")
    else:
        print("No negative cycle present, ok to proceed")

    # Method 1: run Bellman-Ford for n times, time complexity = O(mn^2)
    start = time.time()
    with Pool(cpu_count() // 2) as p:
        result = p.map(partial(bellman_ford, adj_list=adjacency_list), range(nb_nodes))
    print(f"Bellman-Ford: shortest shortest distance = {np.min(np.array(result))}, time consumed = {round(time.time() - start)}s")

    # Method 2: run Floyd_Warshall 1 time, time complexity = O(n^3)
    # Preprocessing
    result_plane = 1000*np.ones((nb_nodes, nb_nodes), dtype=np.dtype('i2'))  # dtype = 16-bit integer for speed & space
    for vertex in range(nb_nodes):
        result_plane[vertex, vertex] = 0
    for tail, head, length in edges:
        result_plane[tail, head] = length
    start = time.time()
    result = floyd_warshall(initial_result_plane=result_plane)
    print(f"Floyd-Warshall: shortest shortest distance = {np.min(result)}, "
          f"time consumed = {round(time.time() - start, 1)}s")

    # Method 3: Johnson's algo, time complexity = O(m*n*logn)
    start = time.time()
    result = johnson(adj_list=adjacency_list, all_edges=edges)
    # Run Dijkstra's algorithm n times.
    # Multiprocessing cannot be used since pickle requires a large recursion depth in this case.
    print(f"Johnson: shortest shortest distance = {np.min(np.array(result))}, "
          f"time consumed = {round(time.time() - start, 1)}s")

    # Conclusion:
    # On large graph, method 1/2 takes a bit more than 6 hours while method 3,
    # even though not optimized for speed,  only takes a bit more than 5 minutes.
