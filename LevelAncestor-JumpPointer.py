## https://www.geeksforgeeks.org/jump-pointer-algorithm/
# https://www.geeksforgeeks.org/level-ancestor-problem/
import math
from node import Node

def constructGraph(jump, depth, node, isNode, c, p, i, R):
    node[i] = c
    if p not in edges:
        edges[p] = []
    edges[p].append(c)

    if isNode[node[i]] == 0:
        isNode[node[i]] = 1
        jump[node[i]][0] = p

        if R == p:
            depth[node[i]] = 1
    return


def compute_len(n):
    return int(math.floor(math.log(n)/math.log(2) + 1))


def set_jump_pointer(jump, node, depth, length, n, R):
    depth[R] = 0
    for j in range(1, length + 1):
        for i in range(n):
            jump[node[i]][j] = jump[jump[node[i]][j - 1]][j - 1]

            if jump[node[i]][j] == R and depth[node[i]] == -1:
                depth[node[i]] = int(math.pow(2, j))
            elif jump[node[i]][j] == 0 and node[i] != R and depth[node[i]] == -1:
                depth[node[i]] = int(pow(2, j - 1) + depth[jump[node[i]][j - 1]])
    for i in range(n):
        print("node %d's height is %d" % (node[i], depth[node[i]]))
    print(depth[:12])


def level_ancestor(jump, isNode, depth, x, L):
    assert isNode[x] == 1, \
        "v must be a valid entry point"
    j, n, k = 0, x, depth[x] - L
    while k > 0:
        if k & 1:
            x = jump[x][j]
        k >>= 1
        j += 1
    print("%d th parent of node %d is %d" % (L, n, x))
    return


n = 12
jump = [[0 for i in range(10)] for i in range(1000)]
depth = [-1] * 1000
node = [0] * 1000
isNode = [0] * 1000
length = compute_len(n)
edges = {}
R = 2
root = Node(R)
constructGraph(jump, depth, node, isNode, 2, 0, 0, R)
constructGraph(jump, depth, node, isNode, 5, 2, 1, R)
constructGraph(jump, depth, node, isNode, 3, 5, 2, R)
constructGraph(jump, depth, node, isNode, 4, 5, 3, R)
constructGraph(jump, depth, node, isNode, 1, 5, 4, R)
constructGraph(jump, depth, node, isNode, 7, 1, 5, R)
constructGraph(jump, depth, node, isNode, 9, 1, 6, R)
constructGraph(jump, depth, node, isNode, 10, 9, 7, R)
constructGraph(jump, depth, node, isNode, 11, 10, 8, R)
constructGraph(jump, depth, node, isNode, 6, 10, 9, R)
constructGraph(jump, depth, node, isNode, 8, 10, 10, R)
set_jump_pointer(jump, node, depth, length, n, R)
level_ancestor(jump, isNode, depth, 2, 0)
level_ancestor(jump, isNode, depth, 4, 1)
level_ancestor(jump, isNode, depth, 5, 1)

level_ancestor(jump, isNode, depth, 10, 1)
level_ancestor(jump, isNode, depth, 11, 1)
level_ancestor(jump, isNode, depth, 6, 1)
level_ancestor(jump, isNode, depth, 8, 1)
print(edges)
print(node)




