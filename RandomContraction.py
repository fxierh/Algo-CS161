import math
import random
from multiprocessing import Pool, cpu_count
import time


# Adjacency list representation of an undirected graph, memory complexity = O(m+n) with m/n = number of edges/vertices.
class AdjacencyList:
    def __init__(self, nb_vertices: int, verbosity: bool = False, parallel_edges: bool = False):
        # Number of vertices. Changes during random contraction.
        self.nb_vertices: int = nb_vertices
        # Number of edges. Changes during random contraction.
        self.nb_edges: int = 0
        # For debugging
        self.verbosity = verbosity
        # Parallel edges allowed in the INITIAL graph or not
        self.parallel_edges = parallel_edges
        # Grouping of vertices (after random contraction there will only be 2 vertex groups left).
        # Changes during random contraction.
        self.vertex_groups: list = [{idx} for idx in range(self.nb_vertices)]
        # List of indices of (remaining) edges. Changes during random contraction.
        self.edge_indices: list = list(range(self.nb_edges))
        # self.vertices[i] contains indices of edges which connect to vertex i, it does not change after initialization.
        self.vertices: list = [set() for _ in range(nb_vertices)]
        # self.edges contains all edges of the form {index of vertex group 1, index of vertex group 2}.
        # It does not change after initialization.
        self.edges: list = []

    # Used by the print function
    def __repr__(self):
        if self.verbosity:
            return f'''\nAdjacency list with {self.nb_vertices} vertices, {self.nb_edges} edges remaining.
Remaining edges = {self.edge_indices}.\nVertex groups = {self.vertex_groups}. '''
        else:
            return f"\nAdjacency list with {self.nb_vertices} vertices, {self.nb_edges} edges."

    def add_edge(self, vertex_1: int, vertex_2: int):
        new_edge = {vertex_1, vertex_2}
        # Parallel edge check
        if new_edge in self.edges and not self.parallel_edges:
            if self.verbosity:
                print(f"Parallel edge not allowed, new edge {new_edge} ignored.")
        else:
            new_edge_idx = self.nb_edges
            self.edges.append(new_edge)
            self.edge_indices.append(self.nb_edges)
            self.nb_edges += 1
            # Sanity check
            # assert new_edge_idx not in self.vertices[vertex_1] and new_edge_idx not in self.vertices[vertex_2]
            self.vertices[vertex_1].add(new_edge_idx)
            self.vertices[vertex_2].add(new_edge_idx)
            if self.verbosity:
                print(f"Edge index {new_edge_idx} added to vertices {vertex_1} and {vertex_2}.")

    # Restore class state after random contraction
    def reinitialization(self):
        self.nb_vertices = len(self.vertices)
        self.nb_edges = len(self.edges)
        self.vertex_groups = [{idx} for idx in range(self.nb_vertices)]
        self.edge_indices = list(range(self.nb_edges))

    # Karger's random contraction algorithm
    def random_contraction(self, random_seed: int or None = None) -> (list, int):
        # Set random seed
        if random_seed is not None:
            random.seed(random_seed)

        while self.nb_vertices > 2:
            # Pick random edge from remaining edges
            edge = self.edges[self.edge_indices[random.randint(0, self.nb_edges - 1)]]

            # Merge vertex groups of the 2 vertices connected by the edge picked
            vertex_1, vertex_2 = edge
            idx_grp_1 = idx_grp_2 = None
            vertex_grp_1 = vertex_grp_2 = None
            for idx_grp, vertex_grp in enumerate(self.vertex_groups):
                if vertex_1 in vertex_grp and not vertex_grp_1:  # Python set membership check is O(1)
                    idx_grp_1 = idx_grp
                    vertex_grp_1 = vertex_grp.copy()
                elif vertex_2 in vertex_grp and not vertex_grp_2:
                    idx_grp_2 = idx_grp
                    vertex_grp_2 = vertex_grp.copy()
            # Sanity check
            assert idx_grp_1 is not None and idx_grp_2 is not None
            self.vertex_groups[idx_grp_1].update(vertex_grp_2)
            new_vertex_grp = self.vertex_groups[idx_grp_1].copy()
            del self.vertex_groups[idx_grp_2]
            if self.verbosity:
                print(f"\nEdge {edge} picked, vertex groups {vertex_grp_1} and {vertex_grp_2} merged.")

            # Remove self-loops and the edge picked
            nb_edge_removed = 0
            for i, idx_edge in enumerate(reversed(self.edge_indices)):
                if self.edges[idx_edge].issubset(new_vertex_grp):
                    del self.edge_indices[self.nb_edges - 1 - i]
                    nb_edge_removed += 1
            if self.verbosity:
                print(f"{nb_edge_removed} edges (picked/self-loop) removed.")

            # Update number of vertices and edges
            self.nb_vertices -= 1
            assert self.nb_vertices == len(self.vertex_groups)  # Sanity check
            self.nb_edges = len(self.edge_indices)
            if self.verbosity:
                print(f"{self.nb_vertices} vertices and {self.nb_edges} edges remaining.")

        # Save results before restoration of class states
        cut_groups = self.vertex_groups
        nb_cut = self.nb_edges

        # Restore class states after random contraction algorithm
        self.reinitialization()

        return cut_groups, nb_cut


if __name__ == '__main__':
    # Test
    # test = AdjacencyList(4, True, False)
    # test.add_edge(0, 1)
    # test.add_edge(0, 2)
    # test.add_edge(1, 2)
    # test.add_edge(2, 3)
    # test.add_edge(1, 3)
    # print(test)
    #
    # random.seed(0)
    # vertex_groups, cut = test.random_contraction()
    # print(f"\nVertex groups = {vertex_groups}, cut = {cut}.")
    # test.reinitialization()
    # print(test)

    # Programming assignment 4
    data = open("AssignmentData/Data_contraction_algo.txt", 'r')
    lines = [[int(i) for i in line.strip().split('	')][1:] for line in data.readlines()]
    nb_lines = len(lines)

    adj_list = AdjacencyList(nb_lines, verbosity=False, parallel_edges=False)
    for idx_1, line in enumerate(lines):
        for idx_2 in line:
            adj_list.add_edge(idx_1, idx_2 - 1)
    print(f"Adjacency list after initialization: {adj_list}")

    # Compute min cut with multiprocessing (about 4 times faster than computation without multiprocessing)
    nb_trials = adj_list.nb_vertices ** 2
    start = time.time()
    with Pool(cpu_count()) as p:
        results = p.map(adj_list.random_contraction, range(nb_trials))
    results = list(zip(*results))
    min_cut = min(results[1])
    print(f"Multiprocessing: time consumed = {time.time() - start}s, min cut = {min_cut}.")

    # Compute min cut without multiprocessing
    start = time.time()
    min_cut = math.inf
    for trial in range(nb_trials):
        vertex_groups, cut = adj_list.random_contraction(random_seed=trial)
        if cut < min_cut:
            min_cut = cut
        # print(f"Trial {trial + 1}: vertex groups = {vertex_groups}, cut = {cut}, min cut = {min_cut}")
    print(f"Single-process code: time consumed = {time.time() - start}s, min cut = {min_cut}.")
