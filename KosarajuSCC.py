import sys
from collections import deque
from collections import Counter
import time

sys.setrecursionlimit(10**4)


class Node:
    def __init__(self, label: int):
        self.label: int = label
        self.neighbors: list = []
        self.neighbors_reversed: list = []
        self.explored: bool = False
        self.leader: Node or None = None

    def add_neighbor(self, neighbor):
        self.neighbors.append(neighbor)

    def add_neighbor_reversed(self, neighbor_reversed):
        self.neighbors_reversed.append(neighbor_reversed)


# Adjacency list representation of a directed graph. Space complexity = O(m + n).
class Graph:
    def __init__(self, nb_vertex: int = 0):
        self.nb_vertex = nb_vertex
        self.vertices: list = []
        # Vertices arranged in increasing finishing times
        self.vertices_increasing_f: list = [None]*nb_vertex
        # Current finishing time
        self.t: int = 0
        # Current leader node
        self.s: Node or None = None

    def add_node(self, node: Node):
        self.vertices.append(node)

    # Depth-first search using recursion.
    # Only use it for small graph. Otherwise, the algorithm can hit Python's max recursion depth (1000).
    def dfs(self, node: Node, first_pass: bool):
        node.explored = True
        if first_pass:
            neighbors = node.neighbors_reversed
        else:
            neighbors = node.neighbors
            node.leader = self.s
        for neighbor in neighbors:
            if not neighbor.explored:
                self.dfs(neighbor, first_pass=first_pass)
        if first_pass:
            self.vertices_increasing_f[self.t] = node
            self.t += 1

    # Depth-first search using stack
    def dfs_stack(self, node: Node, first_pass: bool):
        node.explored = True
        stack = deque()  # Stack
        stack.append(node)
        current_node_arr = []
        while stack:
            v = stack.pop()  # v is the current node
            if not first_pass:
                v.leader = self.s
            current_node_arr.append(v)
            neighbors = v.neighbors_reversed if first_pass else v.neighbors
            for neighbor in neighbors:
                if not neighbor.explored:
                    neighbor.explored = True
                    stack.append(neighbor)
        if first_pass:
            nb_nodes_explored = len(current_node_arr)
            self.vertices_increasing_f[self.t:self.t + nb_nodes_explored] = reversed(current_node_arr)
            self.t += nb_nodes_explored

    # First pass = False means this is the second pass
    def dfs_loop(self, first_pass: bool, recursion_version: bool = False):
        if first_pass:
            self.t = 0
            vertices = self.vertices
        else:
            self.s = None
            vertices = self.vertices_increasing_f

        for node in reversed(vertices):
            if not node.explored:
                self.s = node
                if recursion_version:
                    self.dfs(node=node, first_pass=first_pass)
                else:
                    self.dfs_stack(node=node, first_pass=first_pass)

    # Restoration of class states
    def restore_states(self):
        self.t = 0
        self.s = None
        for node in self.vertices:
            node.explored = False
            node.leader = None

    # Kosaraju's algorithm to compute SCCs.
    # Time complexity = O(m + n) where m is the number of vertices and n is the number of nodes.
    def kosaraju(self, recursion_dfs: bool = False):
        self.dfs_loop(first_pass=True, recursion_version=recursion_dfs)
        self.restore_states()
        self.dfs_loop(first_pass=False, recursion_version=recursion_dfs)
        leader_arr = []
        for node in self.vertices:
            leader_arr.append(node.leader.label)
        self.restore_states()
        return leader_arr


if __name__ == '__main__':
    # Code for testing
    nb_nodes = 9
    graph = Graph(nb_nodes)
    for i in range(nb_nodes):
        graph.add_node(Node(i))
    graph.vertices[0].neighbors.append(graph.vertices[8])
    graph.vertices[8].neighbors_reversed.append(graph.vertices[0])
    graph.vertices[8].neighbors.append(graph.vertices[7])
    graph.vertices[7].neighbors_reversed.append(graph.vertices[8])
    graph.vertices[7].neighbors.append(graph.vertices[0])
    graph.vertices[0].neighbors_reversed.append(graph.vertices[7])
    graph.vertices[7].neighbors.append(graph.vertices[5])
    graph.vertices[5].neighbors_reversed.append(graph.vertices[7])
    graph.vertices[5].neighbors.append(graph.vertices[4])
    graph.vertices[4].neighbors_reversed.append(graph.vertices[5])
    graph.vertices[6].neighbors.append(graph.vertices[5])
    graph.vertices[5].neighbors_reversed.append(graph.vertices[6])
    graph.vertices[4].neighbors.append(graph.vertices[6])
    graph.vertices[6].neighbors_reversed.append(graph.vertices[4])
    graph.vertices[4].neighbors.append(graph.vertices[2])
    graph.vertices[2].neighbors_reversed.append(graph.vertices[4])
    graph.vertices[2].neighbors.append(graph.vertices[1])
    graph.vertices[1].neighbors_reversed.append(graph.vertices[2])
    graph.vertices[1].neighbors.append(graph.vertices[3])
    graph.vertices[3].neighbors_reversed.append(graph.vertices[1])
    graph.vertices[3].neighbors.append(graph.vertices[2])
    graph.vertices[2].neighbors_reversed.append(graph.vertices[3])

    leaders = graph.kosaraju(recursion_dfs=True)
    print(f"Test code with recursion version of DFS: leaders = {leaders}")

    # Initialization
    nb_nodes = 875714
    graph = Graph(nb_nodes)
    for i in range(nb_nodes):
        graph.add_node(Node(i))
    with open("AssignmentData/Data_SCC.txt", 'r') as f:
        for line in f:
            data = line.strip().split(' ')
            src_node, dst_node = int(data[0]) - 1, int(data[1]) - 1
            graph.vertices[src_node].neighbors.append(graph.vertices[dst_node])
            graph.vertices[dst_node].neighbors_reversed.append(graph.vertices[src_node])

    start = time.time()
    leaders = graph.kosaraju(recursion_dfs=False)
    print(f"Programming assignment: time consumed = {round(time.time() - start, 2)}s")
    print(f"Programming assignment with stack version of DFS: 5 largest SCCs = {Counter(leaders).most_common(5)}")
