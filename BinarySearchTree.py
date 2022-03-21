import random


# Class for tree nodes.
class Node:
    def __init__(self, key, left_child=None, right_child=None, parent=None, subtree_size: int = 1):
        self.key = key
        self.left_child = left_child
        self.right_child = right_child
        self.parent = parent
        # Data structure augmentation: size of subtree starting at this node
        self.subtree_size: int or None = subtree_size


# Class for unbalanced binary search trees
class BinarySearchTree:
    def __init__(self, root: Node):
        self.root: Node = root

    # If multiple match, return the one that is closer to the root. Time complexity = O(log(n)) if tree balanced.
    def search(self, key) -> Node or None:
        current_node = self.root
        while current_node:
            current_key = current_node.key
            if key < current_key:
                current_node = current_node.left_child
            elif key == current_key:
                return current_node
            else:
                current_node = current_node.right_child
        return current_node

    # Time complexity = O(log(n)) if tree balanced.
    def insert(self, node: Node):
        current_node = self.root
        key = node.key
        while True:
            current_node.subtree_size += 1
            if key < current_node.key:
                if current_node.left_child:
                    current_node = current_node.left_child
                else:
                    current_node.left_child = Node(key=key, parent=current_node)
                    return
            else:
                if current_node.right_child:
                    current_node = current_node.right_child
                else:
                    current_node.right_child = Node(key=key, parent=current_node)
                    return

    # Return min/max key. Time complexity = O(log(n)) if tree balanced.
    @staticmethod
    def subtree_extremum(root: Node, find_max: bool) -> Node or None:
        current_node = root
        while True:
            if find_max:
                next_node = current_node.right_child
            else:
                next_node = current_node.left_child
            if next_node:
                current_node = next_node
            else:
                return current_node

    # Time complexity = O(log(n)) if tree balanced.
    def pred(self, key) -> Node or None:
        node = self.search(key=key)
        assert node, f"Element with key = {key} is not found in the tree!"
        if node.left_child:
            return self.subtree_extremum(root=node.left_child, find_max=True)
        else:
            while node.parent:
                if node.parent.key > key:
                    node = node.parent
                else:
                    return node.parent
            return None

    # Time complexity = O(log(n)) if tree balanced.
    def succ(self, key) -> Node or None:
        node = self.search(key=key)
        assert node, f"Element with key = {key} is not found in the tree!"
        if node.right_child:
            return self.subtree_extremum(root=node.right_child, find_max=False)
        else:
            while node.parent:
                if node.parent.key < key:
                    node = node.parent
                else:
                    return node.parent
            return None

    # Time complexity = O(n) whether tree balanced or not.
    def __repr__(self):
        # print_key = True -> print keys in order, print_key = False -> print subtree sizes in (subtree keys') order
        def subtree_nodes_in_order(root: Node, print_key: bool) -> str:
            string = f""
            if root.left_child:
                string = subtree_nodes_in_order(root.left_child, print_key=print_key) + string
            if print_key:
                string += f"{root.key} "
            else:
                string += f"{root.subtree_size} "
            if root.right_child:
                string += subtree_nodes_in_order(root.right_child, print_key=print_key)
            return string

        return subtree_nodes_in_order(root=self.root, print_key=True) + f"\t\t" \
            + subtree_nodes_in_order(root=self.root, print_key=False)

    # Swap two node's positions. Time complexity = O(1).
    @staticmethod
    def swap(node1, node2):
        # Attributes of class Node (part from the built-in ones, the pointers and the subtree size)
        attributes = [attr for attr in dir(node1)
                      if not attr.startswith('__')
                      and attr not in ["parent", "left_child", "right_child", "subtree_size"]]
        # Swap previously defined attributes
        for attr in attributes:
            temp1 = node1.__getattribute__(attr)
            temp2 = node2.__getattribute__(attr)
            setattr(node1, attr, temp2)
            setattr(node2, attr, temp1)

    # Remove node with a certain key and return it. Time complexity = O(logn) if tree is balanced.
    def pop(self, key) -> Node:
        node = self.search(key=key)
        assert node, f"Element with key = {key} is not found in the tree!"
        # If a node does not have child, simply remove it from tree.
        if not node.left_child and not node.right_child:
            parent = node.parent
            if parent:
                if parent.left_child == node:
                    parent.left_child = None
                else:
                    parent.right_child = None
        # If a node has both children, it must have a predecessor. Swap node with predecessor then pop swapped node.
        elif node.left_child and node.right_child:
            predecessor = self.pred(key=key)
            # Swap node and predecessor's positions
            self.swap(node, predecessor)
            # Now pop the swapped node
            parent = predecessor.parent
            if predecessor.left_child:  # predecessor does not have a right child as node has both child
                predecessor.left_child.parent = parent
            if parent.left_child == predecessor:
                parent.left_child = predecessor.left_child if predecessor.left_child else None
            else:  # parent.right_child == predecessor
                parent.right_child = predecessor.left_child if predecessor.left_child else None
            node = predecessor
        # If a node has 1 child, pop the node and let its child take its position.
        else:
            child = node.left_child if node.left_child else node.right_child
            parent = node.parent
            if parent:
                child.parent = parent
                if parent.left_child == node:
                    parent.left_child = child
                else:
                    parent.right_child = child
            else:
                child.parent = None
        # Adjust the subtree_size attribute of each node
        while parent:
            parent.subtree_size -= 1
            parent = parent.parent
        # Reinitialize node to return
        node.left_child = node.right_child = node.parent = None
        node.subtree_size = 1
        return node

    # Return the ith smallest element of subtree. Time complexity = O(logn) if tree is balanced.
    def select(self, root: Node, order_statistic: int) -> Node:
        assert 1 <= order_statistic <= self.root.subtree_size, "Order statistic impossible!"

        left_subtree_size = 0
        if root.left_child:
            left_subtree_size = root.left_child.subtree_size

        if order_statistic == left_subtree_size + 1:
            return root
        elif order_statistic < left_subtree_size + 1:
            return self.select(root=root.left_child, order_statistic=order_statistic)
        else:  # order_statistic > left_subtree_size + 1
            return self.select(root=root.right_child, order_statistic=order_statistic - left_subtree_size - 1)

    # Time complexity = O(logn) if tree is balanced. Here rank = order statistic of key.
    # If multiple match, return the smallest rank.
    def rank(self, key) -> int or None:
        current_node = self.root
        rank = 0
        while current_node:
            current_key = current_node.key
            if key < current_key:
                current_node = current_node.left_child
            elif key == current_key:
                return rank + 1 + current_node.left_child.subtree_size if current_node.left_child else rank + 1
            else:  # key > current key
                rank += current_node.left_child.subtree_size + 1 if current_node.left_child else 1
                current_node = current_node.right_child
        return None


if __name__ == '__main__':
    # Constructor test
    max_key = 100
    random.seed(0)
    rt = Node(key=random.randint(0, max_key))
    bts = BinarySearchTree(root=rt)
    print(f"Initial binary search tree = {bts}\n")

    # Test insert method
    nb_nodes_to_add = 9
    keys = []
    for i in range(nb_nodes_to_add):
        new_key = random.randint(0, max_key)
        keys.append(new_key)
        nd = Node(key=new_key)
        bts.insert(node=nd)
        print(f"Inserting node with key {new_key}")
        print(f"Binary search tree = {bts}")

    # Test search method
    print()
    for k in keys:
        print(f"Searching key {k} (in tree: {k in keys})")
        nd = bts.search(key=k)
        print(f"Result = {nd.key if nd else nd}")
    k = max_key + 1
    print(f"Searching key {k} (in tree: {k in keys})")
    nd = bts.search(key=k)
    print(f"Result = {nd.key if nd else nd}")

    # Test subtree_extremum method
    print()
    print(f"Tree is {bts}")
    print(f"Maximum of tree = {bts.subtree_extremum(root=bts.root, find_max=True).key}")
    print(f"Minimum of tree = {bts.subtree_extremum(root=bts.root, find_max=False).key}")

    # Test pred and succ methods
    print()
    print(f"Tree is {bts}")
    for k in keys:
        print(f"Predecessor of key {k} is {bts.pred(key=k).key if bts.pred(key=k) else bts.pred(key=k)}")
        print(f"Successor of key {k} is {bts.succ(key=k).key if bts.succ(key=k) else bts.succ(key=k)}")

    # Test swap method
    print()
    print(f"Tree is {bts}")
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            nd1 = bts.search(keys[i])
            nd2 = bts.search(keys[j])
            bts.swap(nd1, nd2)
            print(f"Swapping keys {nd1.key} and {nd2.key}, resulting tree is {bts}")
            bts.swap(nd1, nd2)
            print(f"Swapping back, resulting tree is {bts}")

    # Test pop method
    print()
    print(f"Tree is {bts}")
    for k in keys:
        nd = bts.pop(key=k)
        print(f"Removing key {k}, popped node has key {nd.key}, resulting tree is {bts}")
    for k in keys:
        bts.insert(node=Node(key=k))
    print(f"Inserting popped keys back, resulting tree is {bts}")

    # Test select method
    print()
    print(f"Tree is {bts}")
    for i in range(1, len(keys) + 2):
        nd = bts.select(root=bts.root, order_statistic=i)
        print(f"Selecting the {i}-th smallest element of tree, resulting key = {nd.key}")

    # Test rank method
    print()
    print(f"Tree is {bts}")
    for k in keys:
        rk = bts.rank(key=k)
        print(f"The rank of key {k} is {rk}")
    k = max_key + 1
    rk = bts.rank(key=k)
    print(f"The rank of key {k} is {rk}")
