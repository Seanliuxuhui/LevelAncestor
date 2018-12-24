# https://github.com/mikolalysenko/level-ancestor/blob/master/level-ancestor.js
import math
import numpy as np
import operator
import numpy as np

R, n = 2, 30
class LadderEntry:
    def __init__(self, ladder, offset):
        self.ladder = ladder
        self.offset = offset


def visit(edges, R, size):
    size = size + 1
    nodes, parents, depths = [-1] * size, [-1] * size, [-1] * size
    visited_nodes = []
    queue = [R]
    depth = 0
    depths[R] = depth
    while queue:
        tmp_queue = []
        while queue:
            w = queue.pop(0)
            visited_nodes.append(w)
            if w in edges:
                for v in edges[w]:
                    tmp_queue.append(v)
                    depths[v] = depth + 1
                    parents[v] = w
                    nodes[w] = w
        depth += 1
        queue = tmp_queue
    return visited_nodes, parents, depths

ladders = np.empty(n, dtype=object)
node_ladders = np.empty(n, dtype=object)

edges = {0: [2], 1: [7, 9, 15], 2: [5], 5: [3, 4, 1], 8: [12], 9: [10], 10: [11, 6, 8], 11: [14], 12: [13], 15: [16, 17], 17: [18], 18: [19]}
R = 2
nodes, parents, depths = visit(edges, R, 20)
print(nodes)
depths_nodes_indexes = np.argsort(depths)[::-1]
for idx in depths_nodes_indexes:
    if depths[idx] == -1:
        break
    path = [idx]
    ladders[idx] = path
    while True:
        x = parents[idx]
        if x < 0:
            break
        idx = x
        ladders[idx] = path
        path.append(idx)
    count = len(path)
    for j in range(count):
        x = parents[idx]
        if x < 0:
            break
        idx = x
        path.append(idx)
    node_ladder = path
    for idx in path:
        node_ladders[idx] = LadderEntry(node_ladder, len(path))

class JumpTable:
    def __init__(self):
        self.jump_table_ = {}
        self.clear = False

    def add_entry(self, node, jumps):
        self.jump_table_[node] = jumps

    def clear(self):
        self.clear = True
        return self.clear

    def renew(self):
        self.jump_table_ = {}

    def get(self, node):
        return -1 if node not in self.jump_table_ else self.jump_table_[node]

def jump_ladder(idx, step, R):
    if step == 0:
        return idx
    ladder = ladders[idx]
    offset = node_ladders[idx].offset
    if step < offset:
        return ladder[step]
    else:
        return -1



jump_table = JumpTable()

for i in nodes:
    jumps = []
    idx = i
    j, k = 1, 0
    while True:
        idx = jump_ladder(idx, j - k, R)
        if idx < 0:
            break
        k = j
        jumps.insert(0, node_ladders[idx])
        j *= 2
    jumps.append(node_ladders[i])
    jump_table.add_entry(i, jumps)

def level_ancestor(node, k):
    if k < 0:
        return -1
    if k == 0:
        return node
    jumps = jump_table.get(node)
    level = int(math.log(k)/math.log(2)) + 1
    if level >= len(jumps):
        return -1
    ladder = jumps[level]
    if k > ladder.offset:
        return -1
    else:
        return ladder.ladder[ladder.offset - k]
print("level ancestor")
for node in [2, 5, 3, 4, 1, 7, 9, 10, 11, 6, 8]:
    for j in range(len(ladders[node])):
        print("%d th parent of node %d is %d"%(j, node, level_ancestor(node, j)))