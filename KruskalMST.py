import numpy as np
import time

from typing import List, Set


class Node:
    def __init__(self, label: int):
        self.label = label
        self.leader = self
        self.rank = 0  # Useful for lazy union find
        self.elements: Set[Node] = {self}  # Store element in component if this node is the leader


# Lazy union-find data structure with rank and path compression, SOTA of union-find design.
# m union + find operations takes O(m*alpha(n)) time, where alpha is the inverse Ackermann function.
class LazyUnionFind:
    def __init__(self, node_array: List[Node]):
        self.node_array = node_array

    def find(self, node_idx: int) -> Node:
        path_compression_nodes = []  # Nodes to apply path compression on
        current_node = self.node_array[node_idx]
        while current_node.leader != current_node:
            path_compression_nodes.append(current_node)
            current_node = current_node.leader
        # Apply path compression
        for node in path_compression_nodes[:-1]:
            node.leader = current_node
        return current_node

    def union(self, node_idx_1: int, node_idx_2: int):
        leader1 = self.find(node_idx_1)
        leader2 = self.find(node_idx_2)
        if leader1.rank > leader2.rank:
            # Make leader 2 points to leader 1
            leader2.leader = leader1
        else:
            # Make leader 1 points to leader 2
            leader1.leader = leader2
            if leader1.rank == leader2.rank:
                leader2.rank += 1

    def __repr__(self):
        leaders = set()
        result = []
        for idx, node in enumerate(self.node_array):
            leader = self.find(idx).label
            leaders.add(leader)
            result.append((node.label, leader, node.rank))
        return f"Union find of {len(leaders)} clusters. Leaders = {leaders}, node_array = {result}"


# Eager union-find data structure with size.
class EagerUnionFind:
    def __init__(self, node_array: List[Node]):
        self.node_array = node_array

    # Return the leader of a node. Time complexity = O(1).
    def find(self, node_idx: int) -> Node:
        return self.node_array[node_idx].leader

    # Merge two components. Time complexity = O(nlogn) for n merge operations, = O(n) for single merge operation.
    def union(self, node_idx_1: int, node_idx_2: int):
        node1 = self.node_array[node_idx_1]
        node2 = self.node_array[node_idx_2]
        if len(node1.leader.elements) < len(node2.leader.elements):
            # Change leader of each node in component 1
            old_leader = node1.leader
            new_leader = node2.leader
        else:
            # Change leader of each node in component 2
            old_leader = node2.leader
            new_leader = node1.leader
        # Update leader's components
        new_leader.elements = new_leader.elements.union(old_leader.elements)
        # Rewire leader pointer
        for node in old_leader.elements:
            node.leader = new_leader

    def __repr__(self):
        leaders = set()
        result = []
        for node in self.node_array:
            leaders.add(node.leader.label)
            result.append((node.label, node.leader.label, len(node.leader.elements)))
        return f"Union find of {len(leaders)} clusters. Leaders = {leaders}, node_array = {result}"


# Kruskal's minimum spanning tree algo used for single linkage clustering.
# Time complexity = O(mlogn) where m = nb of edges, n = nb of vertices.
def kruskal_mst(sorted_edges: np.ndarray, union_find: EagerUnionFind or LazyUnionFind, nb_clusters: int = 1) -> int:
    current_nb_clusters = len(union_find.node_array)
    for idx, edge in enumerate(sorted_edges):
        node1_index, node2_index, e_cost = edge
        if union_find.find(node1_index).label != union_find.find(node2_index).label:
            print(f"Merging edge {node1_index}-{node2_index} of cost = {e_cost}, "
                  f"current number of clusters = {current_nb_clusters}")
            union_find.union(node1_index, node2_index)
            current_nb_clusters -= 1
            if current_nb_clusters == nb_clusters - 1:
                spacing = sorted_edges[idx][-1]
                return spacing


def kruskal_mst2(vertex_dict: dict, union_find: EagerUnionFind or LazyUnionFind, target_min_spacing: int = 3) -> int:
    current_nb_clusters = len(union_find.node_array)
    node_pairs_to_connect: List[set] = []
    for spacing in range(1, target_min_spacing, 1):
        for node_number, node_idx in enumerate(vertex_dict.keys()):
            # print(f"Spacing = {spacing}, node index = {node_idx}, node number = {node_number}")
            combinations = n_choose_k_comb(24, k=spacing)
            for combination in combinations:
                # Compute a potential node with spacing = spacing
                neighbor = list(node_idx)
                for bit_idx in combination:
                    neighbor[bit_idx] = '0' if neighbor[bit_idx] == '1' else '1'
                neighbor = ''.join(neighbor)  # Convert back to string from list
                # If the potential node exists
                if neighbor in vertex_dict:
                    node_pairs_to_connect.append({node_number, vertex_dict[neighbor]})
    # Remove duplicates
    node_pairs_to_connect = [set(item) for item in set(frozenset(item) for item in node_pairs_to_connect)]
    print(f"{len(node_pairs_to_connect)} node pairs to connect")
    # Connect node pairs
    for progress, node_pair in enumerate(node_pairs_to_connect):
        idx1, idx2 = node_pair
        if union_find.find(idx1).label != union_find.find(idx2).label:
            union_find.union(idx1, idx2)
            current_nb_clusters -= 1
        if progress % 10000 == 0:
            print(f"Progress = {progress}")
    return current_nb_clusters


# n_choose_k_comb(5, 2) = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
def n_choose_k_comb(n: int, k: int) -> List[tuple]:
    all_nbs = set(range(n))
    combinations = [(x, ) for x in range(n)]
    for _ in range(1, k, 1):
        temp = []
        for combination in combinations:
            max_value = max(combination)
            new_choices = [nb for nb in all_nbs - set(combination) if nb > max_value]
            for new_choice in new_choices:
                temp.append(combination + (new_choice, ))
        combinations = temp
    return combinations


if __name__ == '__main__':
    # Part 1
    print("Part 1 starts")
    nb_nodes = 500
    edges = []  # [(edge 1 node 1 index, edge 1 node 2 index, edge 1 cost), (...), ...]
    with open("AssignmentData/Data_clustering.txt", 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            else:
                node1_idx, node2_idx, edge_cost = line.strip().split(' ')
                node1_idx, node2_idx, edge_cost = int(node1_idx), int(node2_idx), int(edge_cost)
                edges.append((node1_idx - 1, node2_idx - 1, edge_cost))
    edges = np.array(edges, dtype=[('nd1', 'i4'), ('nd2', 'i4'), ('cost', 'i4')])
    edges = np.sort(edges, order='cost')  # Sort edges by edge cost

    nodes = []
    for i in range(nb_nodes):
        nodes.append(Node(label=i))
    # uf = EagerUnionFind(node_array=nodes)
    uf = LazyUnionFind(node_array=nodes)
    start = time.time()
    min_cluster_spacing = kruskal_mst(edges, uf, nb_clusters=4)
    print(f"Result max spacing = {min_cluster_spacing}, time consumed = {round(time.time() - start, 2)}s")
    print(uf)
    # input("Type anything to continue part 2\n")

    # Part 2
    print("Part 2 starts")
    nb_nodes = 200000
    n_bits = 24
    vertices = dict()
    nodes = []
    with open("AssignmentData/Data_clustering_big.txt", 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            else:
                vertex = "".join(line.strip().split(' '))
                vertices[vertex] = None
    for i, vertex in enumerate(vertices.keys()):
        nodes.append(Node(label=i))
        vertices[vertex] = i
    # uf = EagerUnionFind(node_array=nodes)
    uf = LazyUnionFind(node_array=nodes)
    start = time.time()
    max_nb_clusters = kruskal_mst2(vertices, uf, target_min_spacing=3)
    print(f"Result max k = {max_nb_clusters}, time consumed = {round(time.time() - start)}s")
