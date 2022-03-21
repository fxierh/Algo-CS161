# Use dynamic programming to compute (weight of) the max weight indep set of a path graph. Time complexity = O(n).
def max_weight_indep_set(v_weights: list) -> list:
    ans = [0]*len(v_weights)  # ans[i] = weight of the max weight indep set of the first i vertices
    ans[0] = 0
    ans[1] = v_weights[1]
    for idx in range(2, len(v_weights), 1):
        ans[idx] = max(ans[idx - 1], ans[idx - 2] + v_weights[idx])
    return ans


def indep_set_reconstruction(ans_list: list, v_weights: list) -> list:
    indep_set = []  # Max weight independent set, containing index of vertices (starting from 1)
    vertex_idx = len(v_weights) - 1
    while vertex_idx >= 1:
        if ans_list[vertex_idx - 1] >= ans_list[vertex_idx - 2] + v_weights[vertex_idx]:
            vertex_idx -= 1
        else:
            indep_set.append(vertex_idx)
            vertex_idx -= 2
    return indep_set


if __name__ == '__main__':
    vertex_weights = [None]  # vertex_weights[i] = weight of vertex i
    with open("AssignmentData/Data_Max_Weight_Indep_Set.txt", 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            else:
                vertex_weights.append(int(line.strip()))
    ans_array = max_weight_indep_set(vertex_weights)
    max_weight_independent_set = indep_set_reconstruction(ans_array, vertex_weights)
    print(f"Max weight independent set = {max_weight_independent_set}")

    vertex_indices = [1, 2, 3, 4, 17, 117, 517, 997]
    string = ""
    for i in vertex_indices:
        if i in max_weight_independent_set:
            string += "1"
        else:
            string += "0"
    print(string)
