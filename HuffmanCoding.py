from queue import Queue
from typing import List


class Node:
    def __init__(self, freq: int, left_child=None, right_child=None, label=None):
        self.label = label  # Only leaf node has a label
        self.freq = freq
        self.left_child = left_child
        self.right_child = right_child


# Iterative greedy Huffman coding with time complexity = O(nlogn).
def huffman_coding(nds: List[Node]) -> Node:
    # Sort nodes by frequencies from largest to smallest, time complexity = O(nlogn).
    nds = sorted(nds, reverse=True, key=lambda x: x.freq)
    q = Queue()
    alphabet_size = len(nds)

    # Time complexity of while loop = O(n)
    while alphabet_size >= 2:
        two_least_frequent_nodes = []
        for _ in range(2):
            if q.empty():
                two_least_frequent_nodes.append(nds.pop())
                continue
            if not nds:
                two_least_frequent_nodes.append(q.get())
                continue
            if q.queue[0].freq > nds[-1].freq:
                two_least_frequent_nodes.append(nds.pop())
            else:
                two_least_frequent_nodes.append(q.get())
        q.put(Node(freq=two_least_frequent_nodes[0].freq + two_least_frequent_nodes[1].freq,
                   left_child=two_least_frequent_nodes[0], right_child=two_least_frequent_nodes[1]))
        alphabet_size -= 1

    # Return root of tree
    root = q.get()
    return root


# In order traversal of tree to get Huffman coding in O(n) time.
def get_coding(root: Node, current_string: str = '') -> dict:
    h_coding = dict()

    # If current node is a leaf node
    if not root.left_child and not root.right_child:
        h_coding[root.label] = current_string
        return h_coding

    if root.left_child:
        h_coding.update(get_coding(root.left_child, current_string=current_string + '0'))
    if root.right_child:
        h_coding.update(get_coding(root.right_child, current_string=current_string + '1'))

    return h_coding


if __name__ == '__main__':
    nodes: List[Node] = []

    # Test code
    # nodes.append(Node(freq=3, label='A'))
    # nodes.append(Node(freq=2, label='B'))
    # nodes.append(Node(freq=6, label='C'))
    # nodes.append(Node(freq=8, label='D'))
    # nodes.append(Node(freq=2, label='E'))
    # nodes.append(Node(freq=6, label='F'))

    # Real data
    with open("AssignmentData/Data_Huffman.txt", 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            else:
                nodes.append(Node(freq=int(line.strip()), label=i))

    root_node = huffman_coding(nodes)
    coding = get_coding(root=root_node)
    # Sort by value
    print(sorted(coding.items(), key=lambda item: int(item[1])))
