import time
import numpy as np

from multiprocessing import Pool, cpu_count
from KosarajuSCC import Node, Graph


# Variation of Papadimitriou's 2SAT algorithm with less time complexity.
def papadimitriou(n_vars: int, clause_array: np.ndarray, variable_dict: dict) -> list or None:
    # Choose random initial assignment
    assignment = np.random.choice(a=[False, True], size=n_vars, replace=True)
    assignment = np.append(assignment, np.logical_not(assignment))
    checked_until = 0
    clause_to_check = set()
    for count in range(2 * n_vars ** 2):
        # if count % 1000 == 0:
        #     print(f"Count = {count}")
        satisfy = True  # Indicate if all clauses are satisfied simultaneously or not
        for count_2, clause in enumerate(clause_array):
            # Only re-check checked clauses that can become false, i.e. those that involve the recently changed variable
            if count_2 > checked_until or count_2 in clause_to_check:
                # If clause not satisfied, choose one related variable randomly and flip its value
                if not np.sum(assignment[clause]):
                    var_to_change = np.random.choice(a=clause) % n_vars
                    temp = assignment[var_to_change]
                    assignment[var_to_change] = not temp
                    assignment[var_to_change + n_vars] = temp
                    satisfy = False
                    # Boundary between checked & unchecked is pushed further, update boundary & re-init clause to check
                    if count_2 - 1 > checked_until:
                        checked_until = count_2 - 1
                        clause_to_check = variable_dict[var_to_change]
                        if checked_until % 1000 == 0:
                            print(f"Checked until = {checked_until}, count = {count}")
                    # Else keep boundary unchanged and add new clauses to checking list
                    else:
                        clause_to_check.update(variable_dict[var_to_change])
                    break
        if satisfy:
            # Sanity check
            for idx, clause in enumerate(clause_array):
                assert np.sum(assignment[clause]), \
                    f"Clause {idx} ({clause_array[idx]}) failed with {assignment[clause]}"
            print("Assignment found, terminating all processes")
            return assignment[:n_vars].tolist()
    return None


def quit_processes():
    # p.terminate()  # kill all pool workers
    pass


if __name__ == '__main__':
    # Method 1: reduce 2SAT to the determination of SCCs.
    datasets = range(1, 7, 1)
    for dataset in datasets:
        print(f"Dataset {dataset} begins:")
        with open(f"AssignmentData/Data_2SAT_{str(dataset)}.txt", 'r') as f:
            for i, line in enumerate(f):
                if i == 0:
                    # Build graph
                    nb_vars = int(line.strip())
                    nb_nodes = 2*nb_vars
                    graph = Graph(nb_nodes)
                    for j in range(nb_nodes):
                        graph.add_node(Node(j))
                else:
                    # Add edges
                    # Variable x1, x2, ..., xn correspond respectively to node 0, 1, ..., n - 1
                    # Variable not x1, not x2, ..., not xn correspond respectively to node n, n + 1, ..., 2n - 1
                    node1, node2 = [int(node) for node in line.strip().split(' ')]
                    # Edge 1: not node 1 -> node 2, edge 2: not node 2 -> node 1
                    edges = [[-node1, node2], [-node2, node1]]
                    for a in range(2):
                        for b in range(2):
                            if edges[a][b] < 0:
                                edges[a][b] = abs(edges[a][b]) + nb_vars - 1
                            else:
                                edges[a][b] -= 1
                    for edge in edges:
                        graph.vertices[edge[0]].neighbors.append(graph.vertices[edge[1]])
                        graph.vertices[edge[1]].neighbors_reversed.append(graph.vertices[edge[0]])

        start = time.time()
        leaders = graph.kosaraju(recursion_dfs=False)
        satisfiable = True
        for var in range(nb_vars):
            if leaders[var] == leaders[var + nb_vars]:
                satisfiable = False
                break
        print(f"Satisfiable = {satisfiable}, time consumed = {round(time.time() - start, 2)}s")

    # Method 2: Papadimitriou's 2-SAT algorithm
    with open(f"AssignmentData/Data_2SAT_6.txt", 'r') as f:
        var_dict = dict()  # Only to suppress automatic code inspection errors
        for i, line in enumerate(f):
            if i == 0:
                nb_vars = int(line.strip())
                clauses = np.empty((nb_vars, 2), dtype=np.dtype('i4'))
                # var_dict: variable (not their negation) index: [indices of clauses that involving this variable]
                var_dict = dict(zip(range(nb_vars), [[] for _ in range(nb_vars)]))
            else:
                # Variable x1, x2, ..., xn correspond respectively to node 0, 1, ..., n - 1
                # Variable not x1, not x2, ..., not xn correspond respectively to node n, n + 1, ..., 2n - 1
                vars_involved = [abs(int(var)) - 1 for var in line.strip().split(' ')]
                var_dict[vars_involved[0]].append(i - 1)
                var_dict[vars_involved[1]].append(i - 1)
                clauses[i - 1] = [abs(int(var)) + nb_vars - 1 if var[0] == '-' else int(var) - 1
                                  for var in line.strip().split(' ')]

        # Transform dict values from list to set, to avoid duplicate later
        for key in var_dict.keys():
            var_dict[key] = set(var_dict[key])

    start = time.time()
    # nb_workers = cpu_count()
    # with Pool(nb_workers) as p:
    #     for i in range(nb_workers):
    #         result = p.apply_async(papadimitriou, args=(nb_vars, clauses, var_dict), callback=quit_processes)
    #     print(f"Workers: {nb_workers}, result = {result.get()}, time consumed = {round(time.time() - start, 2)}s")
    result = papadimitriou(nb_vars, clauses, var_dict)
    print(f"Result = {result}, time consumed = {round(time.time() - start, 2)}s")
