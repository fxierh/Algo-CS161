import time
from DijkstraShortestPath import Node, Heap, Graph


# Inherited from the class Node
class NodeMST(Node):
    def __init__(self, label: int, neighbors=None, key: int = 1000000, shortest_distance: int = 1000000,
                 processed: bool = False, heap_index: int or None = None):
        super().__init__(label, neighbors, key, shortest_distance, processed, heap_index)
        # Cheapest edge among those between a processed node and this node. Form: (V1, V2, distance between V1 and V2)
        self.cheapest_edge = None


# Inherited from the class Graph
class GraphMST(Graph):
    def __init__(self, n_nodes: int):
        super().__init__(n_nodes)
        self.mst = []  # Minimum spanning tree
        self.mst_cost: int = 0  # Total edge length of the minimum spanning tree

    # Prim's minimum spanning tree algo, time complexity = O(mlogn) where m = nb of edges, n = nb of vertices.
    # Algo starts from random vertex (the one that happens to be the first element in heap)
    def prim_mst(self, heap: Heap):
        while len(self.processed_nodes) < self.n_nodes:
            new_node = heap.pop()
            new_node.processed = True
            self.processed_nodes.append(new_node)
            print(f"Processing the {len(self.processed_nodes)}th node:")
            # Update MST (ignore first node)
            if new_node.cheapest_edge:
                self.mst.append(new_node.cheapest_edge)
                self.mst_cost += new_node.key
            for neighbor in new_node.neighbors.keys():
                # If neighbor of new node is in heap
                if neighbor.heap_index is not None:
                    print(f"Updating neighbor {neighbor.label}'s key")
                    # Delete neighbor from heap
                    heap.pop(index=neighbor.heap_index)
                    # Update neighbor's key and cheapest edge
                    if new_node.neighbors[neighbor] < neighbor.key:
                        neighbor.key = new_node.neighbors[neighbor]
                        neighbor.cheapest_edge = (new_node.label, neighbor.label, neighbor.key)
                    # neighbor.key = min(neighbor.key, new_node.neighbors[neighbor])
                    # Re-insert neighbor into heap
                    heap.insert(neighbor)


if __name__ == '__main__':
    # Initialization
    graph = GraphMST(n_nodes=500)
    node_array = []
    for i in range(500):
        node_array.append(NodeMST(label=i))
    graph.nodes = node_array
    graph.n_nodes = 500

    with open("AssignmentData/Data_PrimMST.txt", 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            else:
                # Add edge to both vertices it connects
                v1, v2, distance = [int(nb) for nb in line.strip().split(' ')]
                graph.nodes[v1 - 1].neighbors[graph.nodes[v2 - 1]] = distance
                graph.nodes[v2 - 1].neighbors[graph.nodes[v1 - 1]] = distance

    h = Heap(arr=graph.nodes)
    # h.heapify()
    print(f"Initial heap: {h}")

    start = time.time()
    graph.prim_mst(heap=h)
    print(f"Time consumed = {round(time.time() - start, 2)}s")
    print(f"MST is {graph.mst}")
    print(f"MST cost = {graph.mst_cost}")
