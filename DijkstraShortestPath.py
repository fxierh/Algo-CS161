from typing import List
import random
import time


class Node:
    def __init__(self, label: int, neighbors=None, key: int = 1000000, shortest_distance: int = 1000000,
                 processed: bool = False, heap_index: int or None = None):
        if neighbors is None:
            neighbors = {}
        self.label: int = label
        self.neighbors: dict = neighbors  # Each element = neighbor node: distance between this node & the neighbor
        self.processed: bool = processed  # Whether the node is already processed
        self.shortest_distance: int = shortest_distance  # Shortest path distance between node 0 and this node
        self.key = key  # Smallest Dijkstra greedy score: the key used in the heap
        self.heap_index: int or None = heap_index  # Position of node in the heap


# Min heap of nodes.
class Heap:
    def __init__(self, arr: List[Node]):
        self.data: List[Node] = arr
        for idx, node in enumerate(self.data):
            node.heap_index = idx

    # Loop version of the heapify function (rearrange an array into a heap). Time complexity = O(n).
    # Recursion can be used as well.
    def heapify(self):
        # For i = (len(self.data) - 2//2) downto 0: considering all nodes that has at least a child
        for node_idx in range((len(self.data) - 2) // 2, -1, -1):
            self.bubble_down(node_idx)

    # Time complexity = O(logn) where n is the number of heap elements.
    def pop(self, index: int = 0) -> Node or None:
        assert index >= 0, f"Index of the element to pop should be non-negative!"
        assert index <= len(self.data) - 1, f"Index too large (out of range)!"
        assert self.data, f"No node to pop!"

        # If the last element is being popped
        if index == len(self.data) - 1:
            popped_node = self.data.pop()
            popped_node.heap_index = None
        else:
            self.data[-1].heap_index = index
            self.data[index], self.data[-1] = self.data[-1], self.data[index]
            popped_node = self.data.pop()
            popped_node.heap_index = None
            # Check if the current node at index=index has a parent/left_child/right_child or not.
            parent_index = (index - 1) // 2 if index != 0 else None
            left_child_index = 2 * index + 1 if 2 * index + 1 < len(self.data) else None
            right_child_index = None
            if left_child_index:
                if left_child_index + 1 < len(self.data):
                    right_child_index = left_child_index + 1
            if parent_index:  # If current node at index=index has a parent
                if self.data[index].key < self.data[parent_index].key:
                    self.bubble_up(index)
            elif left_child_index:  # If current node at index=index has at least a child
                min_child = min(self.data[left_child_index].key, self.data[right_child_index].key) \
                    if right_child_index else self.data[left_child_index].key
                if self.data[index].key > min_child:
                    self.bubble_down(index)
        return popped_node

    # Bubble the element at a certain index up to the right position. Hypothesis: parent exists.
    def bubble_up(self, index: int):
        parent_index = (index - 1) // 2
        while self.data[index].key < self.data[parent_index].key:
            # Swap index with parent
            self.data[index].heap_index, self.data[parent_index].heap_index = parent_index, index
            # Swap with parent
            self.data[index], self.data[parent_index] = self.data[parent_index], self.data[index]
            # If still has a parent
            if parent_index != 0:
                index = parent_index
                parent_index = (parent_index - 1) // 2
            else:
                break

    # Bubble the element at a certain index down to the right position.
    def bubble_down(self, index: int):
        left_child_index = 2 * index + 1
        right_child_index = left_child_index + 1 if left_child_index + 1 < len(self.data) else None
        left_child_key = self.data[left_child_index].key
        right_child_key = self.data[right_child_index].key if right_child_index else None
        min_child = min(left_child_key, right_child_key) if right_child_index else left_child_key
        while self.data[index].key > min_child:
            if left_child_key == min_child:
                self.data[index].heap_index, self.data[left_child_index].heap_index = left_child_index, index
                self.data[index], self.data[left_child_index] = self.data[left_child_index], self.data[index]
                index = left_child_index
            else:
                self.data[index].heap_index, self.data[right_child_index].heap_index = right_child_index, index
                self.data[index], self.data[right_child_index] = self.data[right_child_index], self.data[index]
                index = right_child_index
            # If still has at least a child
            if 2 * index + 1 < len(self.data):
                left_child_index = 2 * index + 1
                right_child_index = left_child_index + 1 if left_child_index + 1 < len(self.data) else None
                left_child_key = self.data[left_child_index].key
                right_child_key = self.data[right_child_index].key if right_child_index else None
                min_child = min(left_child_key, right_child_key) if right_child_index else left_child_key
            else:
                break

    # Time complexity = O(logn) where n is the number of heap elements.
    def insert(self, node: Node):
        self.data.append(node)
        node.heap_index = len(self.data) - 1
        # Put the newly added node into its rightful position
        self.bubble_up(index=node.heap_index)

    def __repr__(self):
        str_keys = f"Heap with keys = "
        str_heap_idx = f"Heap indices = "
        for idx, node in enumerate(self.data):
            assert idx == node.heap_index
            str_keys += f"{node.key} "
            str_heap_idx += f"{node.heap_index} "
        # return str_keys + "\n" + str_heap_idx
        return str_keys


# Directed graph with non-negative edge lengths
class Graph:
    def __init__(self, n_nodes: int):
        self.processed_nodes: List[Node] = []
        self.nodes: list = []  # List of Node objects
        self.n_nodes: int = n_nodes

    # Dijkstra's shortest path algorithm, time complexity = O(m*logn) where m = number of edges, n = number of nodes.
    def dijkstra(self, heap: Heap):
        while len(self.processed_nodes) < len(self.nodes):
            new_node = heap.pop()
            new_node.shortest_distance = new_node.key
            new_node.processed = True
            self.processed_nodes.append(new_node)
            print(f"Processing the {len(self.processed_nodes)}th node:")
            print(f"Adding node {new_node.label} of shortest distance {new_node.shortest_distance}:")
            for neighbor in new_node.neighbors.keys():
                # If neighbor of new node is in heap
                if neighbor.heap_index is not None:
                    print(f"Updating neighbor {neighbor.label}'s key")
                    # Delete neighbor from heap
                    heap.pop(index=neighbor.heap_index)
                    # Recompute neighbor's key
                    neighbor.key = min(neighbor.key, new_node.shortest_distance + new_node.neighbors[neighbor])
                    # Re-insert neighbor into heap
                    heap.insert(neighbor)


if __name__ == '__main__':
    '''
    # Heap test
    # h = Heap([])
    # nb_node = 15
    # node_array = []
    # random.seed(0)
    # for _ in range(nb_node):
    #     nb = random.randint(0, 10)
    #     # print(f"New key = {nb}")
    #     nd = Node(label=nb, key=nb, neighbors=[])
    #     node_array.append(nd)
    #     h.insert(node=nd)
    # print(h)
    #
    # nd = h.pop(index=9)
    # print(h)
    #
    # h2 = Heap(arr=node_array)
    # h2.heapify()
    # print(h2)
    '''

    # Initialization
    graph = Graph(n_nodes=200)
    node_array = []
    for i in range(200):
        node_array.append(Node(label=i))
    graph.nodes = node_array

    with open("AssignmentData/Data_Dijkstra.txt", 'r') as f:
        for i, line in enumerate(f):
            nbrs = line.strip().split('	')[1:]
            nbrs = dict([tuple((node_array[int(i.split(',')[0]) - 1], int(i.split(',')[1]))) for i in nbrs])
            node_array[i].neighbors = nbrs
            # print(nbrs)
            if i == 0:
                node_array[i].key = 0
                node_array[i].shortest_distance = 0
                node_array[i].processed = True
                graph.processed_nodes.append(node_array[i])
            # If node i is a neighbor of node 0
            elif i in [node.label for node in graph.processed_nodes[0].neighbors.keys()]:
                node_array[i].key = graph.processed_nodes[0].neighbors[node_array[i]]

    h = Heap(arr=node_array[1:])
    h.heapify()
    print(f"Initial heap: {h}")

    start = time.time()
    graph.dijkstra(heap=h)
    time_consumed = round(time.time() - start, 2)
    node_indices = [7, 37, 59, 82, 99, 115, 133, 165, 188, 197]
    shortest_distances = []
    for i in range(200):
        print(f"Node {graph.nodes[i].label}'s shortest distance = {graph.nodes[i].shortest_distance}")
        if i + 1 in node_indices:
            shortest_distances.append(graph.nodes[i].shortest_distance)
    print(shortest_distances)
    print(f"Time consumed = {time_consumed}s")
