import math
import collections
import copy
class StaticMacroMicroTree:
    def __init__(self, R, n, maxlength):
        self.maxlength = maxlength
        self.edges = collections.defaultdict(list)
        self.jump = [[-1] * int(math.log(n, 2)) for i in range(n)]
        self.node = [-1] * n
        self.isnode = [-1] * n
        self.depth = [-1] * n
        self.maxdepth = math.log(maxlength, 2)
        self.size = [-1] * n
        self.froms = [-1] * n
        self.r0 = int(math.log(math.log(n, 2), 2))
        self.n = n
        self.M = int(math.log(n, 2)*0.5)
        self.ranks = [-1] * self.n
        self.R = R
        self.microtree_root_table = [-1] * self.n
        self.node_belongs_to = [-1] * self.n
        self.tree_node_visit_sequence = []
        self.microset_size = collections.defaultdict(int)

    def constructGraph(self, c, p, node_id, R):
        self.node[node_id] = c
        self.edges[p].append(c)
        if self.isnode[self.node[node_id]] == -1:
            self.isnode[self.node[node_id]] = 1
            self.jump[self.node[node_id]][0] = p

            if c == R:
                self.depth[self.node[node_id]] = 0

            if p == R:
                self.depth[self.node[node_id]] = 1
        return
    def build_datastructure(self):
        self._dfs(self.R)
        self.d = collections.defaultdict(int)
        print(self.edges)
        self.froms[self.R] = self.R
        self.jump, self.levelanc, self.macronodes,self.depth_macro_tree = self._preprocess_macro_trees()
        self.jumpM, self.levelancM = self._process_jumpM()
        self.jump_micro = self._dfs_micro_tree()
        print(self.microset_size)

    def rank(self, v, depth, size, maxdepth):
        if depth[v] == 0:
            return int(math.floor(math.log(self.n, 2)))

        idx = 0
        while idx < maxdepth:
            if depth[v] % int(math.pow(2, idx)) == 0 and size[v] >= math.pow(2, idx):
                idx += 1
            else:
                return idx

    def _dfs(self, v):
        children_visited = collections.defaultdict(int)
        stack = [v]
        self.depth[v] = 0
        self.size[v] = 1
        while stack:
            v = stack[-1]
            if v in self.edges:
                if children_visited[v] < len(self.edges[v]):
                    w = self.edges[v][children_visited[v]]
                    self.size[w] = 1
                    self.froms[w] = v
                    self.depth[w] = self.depth[v] + 1
                    stack.append(w)
                    children_visited[v] += 1
                else:
                    item_pop = stack.pop()
                    self.tree_node_visit_sequence.append(item_pop)
                    self.size[self.froms[item_pop]] += self.size[item_pop]
            else:
                item_pop = stack.pop()
                self.tree_node_visit_sequence.append(item_pop)
                self.size[self.froms[item_pop]] += self.size[item_pop]


    def _construct_macro_tree(self, jump, depth, node, isnode, edges, c, p, node_idx, R):
        node[node_idx] = c
        edges[p].append(c)
        if isnode[node[node_idx]] == -1:
            isnode[node[node_idx]] = 1
            jump[node[node_idx]].append(p)

            if R == p:
                depth[node[node_idx]] = 1
        return
    def _dfs_macrotree(self, v, depth, size, edges, parents):
        size[v] = 1
        stack = [v]
        depth[v] = 0
        children_visited = collections.defaultdict(int)
        while stack:
            v = stack[-1]
            if v in edges:
                if children_visited[v] < len(edges[v]):
                    w = edges[v][children_visited[v]]
                    size[w] = 1
                    parents[w] = v
                    depth[w] = depth[v] + 1
                    children_visited[v] += 1
                else:
                    item_pop = stack.pop()
                    size[parents[item_pop]] += size[item_pop]
            else:
                item_pop = stack.pop()
                size[parents[item_pop]] += size[item_pop]
        return
    def _preprocess_macro_trees(self):
        jump = [[-1] * int(math.log(n, 2)) for i in range(self.n)]
        node = [-1] * self.n
        depth = [0] * self.n
        isnode = [-1] * self.n
        edges = collections.defaultdict(list)
        parents = [-1] * self.n
        size = [0] * self.n
        macronodes = [-1] * self.n
        z = []
        ### Step 1: calculate ranks for each node
        for i in range(1, self.n):
            self.ranks[i] = self.rank(i, self.depth, self.size, self.maxdepth)
            macronodes[i] = 1 if self.ranks[i] >= self.r0 else -1
            if macronodes[i] == 1:
                z.append(i)
        print("macronodes:", z)
        ### Step 2: retrieve macro node
        stack = [self.R]
        node_idx = 0
        while stack:
            v = stack.pop(0)
            cur_node = v
            if self.ranks[v] >= self.r0:
                v = self.froms[v]
                while v != -1 and v != self.R and self.ranks[v] < self.r0:
                    v = self.froms[v]
                if cur_node == self.R:
                    v = 0
                if v != -1 and v != cur_node:
                    self._construct_macro_tree(jump, depth, node, isnode, edges, cur_node, v, node_idx, self.R)
                    node_idx += 1

            if cur_node in self.edges:
                stack = self.edges[cur_node] + stack

        ### Step 3: traverse the macro tree and calculate depth
        stack = [node[0]]
        while stack:
            v = stack.pop(0)
            if v in edges:
                for w in edges[v]:
                    depth[w] = depth[v] + 1
                    parents[w] = v
                stack = edges[v] + stack
        # self._dfs_macrotree(node[0], depth, size, edges, parents)

        ### Step 4: calculate ranks for each macro node
        ranks_macro_node = collections.defaultdict(int)
        for i in range(1, self.n):
            ranks_macro_node[i] = self.rank(i, depth, size, self.maxdepth)

        ### Step 5: calculate levelanc for macro trees
        levelanc = {node[i]: [-1] * depth[node[i]] for i in range(node_idx)}
        for i in range(node_idx):
            v = node[i]
            if v != -1 and parents[v] != -1:
                for j in range(depth[node[i]]):
                    levelanc[node[i]][j] = parents[v]
                    v = parents[v]
        levelanc[self.R].append(self.R)
        ### Step 6: build jump table for macro trees
        jump = collections.defaultdict(list)
        R = 0
        stack = [node[0]]
        while stack:
            v = stack.pop(0)
            for i in range(int(math.log(depth[v] + 1, 2))):
                if depth[parents[v]] % int(math.pow(2, i)) == 0:
                    jump[v].append(parents[v])
                else:
                    if i >= len(jump[parents[v]]):
                        jump[v].append(jump[parents[v]][-1])
                    else:
                        jump[v].append(jump[parents[v]][i])
            if v in edges:
                stack = edges[v] + stack
        print(edges)
        return jump, levelanc, macronodes, depth

    def _process_jumpM(self):
        jumpM = [-1] * self.n
        levelancM = collections.defaultdict(list)
        stack = [self.R]
        while stack:
            v = stack.pop(0)
            cur, idx = v, 1
            levelancM[cur] = [cur]
            while v != -1:
                if self.depth[self.froms[v]] % self.M == 0:
                    jumpM[cur] = self.froms[v]
                    break
                # if len(levelancM[cur]) >= self.M:
                #     break
                v = self.froms[v]
            v = cur
            while v != -1 and len(levelancM[cur]) <= self.M:
                levelancM[cur].append(self.froms[v])
                v = self.froms[v]


            if cur in self.edges:
                stack = self.edges[cur] + stack
        jumpM[self.R] = self.R
        return jumpM, levelancM
    def _dfs_micro_tree(self):
        self.microset_idx = 0
        jump = collections.defaultdict(list)
        for node in self.tree_node_visit_sequence:
            self.d[node] = 1
            if node in self.edges:
                ## first find the complete subtree with root of X, for the rest children of X, just let themselves form a tree
                for w in self.edges[node]:
                    if self.d[node] + self.d[w] > (self.M + 1) / 2:
                        self.microtree_root_table[self.microset_idx] = node
                        stack = []
                        for v in self.edges[node]:
                            if v == w:
                                break
                            stack.insert(0, v)
                            jump[v] = [v, node]
                            self.edges[node].remove(v)
                        stack.insert(0, w)
                        jump[w] = [w, node]
                        self.edges[node].remove(w)

                        while stack:
                            v = stack.pop(0)
                            self.node_belongs_to[v] = self.microset_idx
                            self.microset_size[self.microset_idx] += 1
                            if v in self.edges:
                                for each_child in self.edges[v]:
                                    jump[each_child] += [each_child, v]
                                    stack.insert(0, each_child)
                                    if len(jump[v]) > 0:
                                        jump[each_child].extend(jump[v][1:])
                                    self.edges[v].remove(each_child)
                        self.d[node] = 1
                        self.microset_idx += 1
                    else:
                        self.d[node] += self.d[w]



        ## deal with last node
        stack = [node]
        self.microtree_root_table[self.microset_idx] = node
        jump[node] = [node]
        while stack:
            v = stack.pop(0)
            self.node_belongs_to[v] = self.microset_idx
            self.microset_size[self.microset_idx] += 1
            if v in self.edges:
                for w in self.edges[v]:
                    jump[w] = [w, v]
                    if len(jump[v]) > 1:
                        jump[w].extend(jump[v][1:])
                    stack.insert(0, w)
                    self.edges[v].remove(w)
        self.microset_idx += 1

        ## deal with unprecess nodes
        for v, edges in self.edges.items():
            if len(edges) > 0:
                self.microtree_root_table[self.microset_idx] = v
                jump[v] = [v]
                stack = [v]
                while stack:
                    w = stack.pop(0)
                    self.node_belongs_to[w] = self.microset_idx
                    self.microset_size[self.microset_idx] += 1
                    if w in self.edges:
                        for child in self.edges[w]:
                            jump[child] = [child, w]
                            if len(jump[v]) > 1:
                                jump[child].extend(jump[v][1:])
                            stack.insert(0, child)
                            self.edges[w].remove(child)
                self.microset_idx += 1
            del self.edges[v]
        ## extend each jump node to include # M of ancestors
        for node, jump_table in jump.items():
            if len(jump_table) < self.M + 1:
                jump_table.extend(jump[jump_table[-1]][1:self.M + 1 - len(jump_table)+1])

        return jump

    def _level_ancestor(self, v, x):
        i = k = int(math.log(x +1, 2))
        d = self.depth_macro_tree[v] - x
        while 2 * (self.depth_macro_tree[v] - d) >= int(math.pow(2, k)):
            if len(self.jump[v]) < i:
              i -= 1
            v = self.jump[v][i-1]
        if self.depth_macro_tree[v] - d >= 1:
            if len(self.levelanc[v]) <= self.depth_macro_tree[v] - d:
                if len(self.levelanc[v]) > 1:
                    return self.levelanc[v][-1], self.levelanc[v][-2]
                else:
                    return self.levelanc[v][-1], v
            return self.levelanc[v][self.depth_macro_tree[v] - d], self.levelanc[v][self.depth_macro_tree[v] - d - 1]
        else:
            return self.levelanc[v][self.depth_macro_tree[v] - d], v
    def level_ancestor(self, v, x):
        d = self.depth[v] - x
        w = self.jumpM[v]
        if w != -1 and self.macronodes[w] == -1:
            w = self.jumpM[w]
        # now w is the first macro node on the path from v to root(v)
        if w != -1 and self.depth[w] > d:
            v, prev = self._level_ancestor(w, int(math.floor((self.depth[w] - d) / self.M)))
            if self.depth[v] < d:
                v = prev
        # now there no macro nodes on the path from v.
        if self.depth[self.microtree_root_table[self.node_belongs_to[v]]] <= d:
            return self._level_ancestor_micro(v, self.depth[v] - d)
        v = self.froms[self.microtree_root_table[self.node_belongs_to[v]]]
        if self.depth[self.microtree_root_table[self.node_belongs_to[v]]] <= d:
            return self._level_ancestor_micro(v, self.depth[v] - d)
        return self.levelancM[self.microtree_root_table[self.node_belongs_to[v]]][self.depth[self.microtree_root_table[self.node_belongs_to[v]]] - d]

    def _level_ancestor_micro(self, v, x):
        return self.jump_micro[v][x]

    def get_depth(self, i):
        return self.depth[i]

n, R = 30, 2
macro_micro_tree = StaticMacroMicroTree(R, n, 3000)

macro_micro_tree.constructGraph(2, 0, 0, R)
macro_micro_tree.constructGraph(5, 2, 1, R)
macro_micro_tree.constructGraph(3, 5, 2, R)
macro_micro_tree.constructGraph(4, 5, 3, R)
macro_micro_tree.constructGraph(1, 5, 4, R)
macro_micro_tree.constructGraph(7, 1, 5, R)
macro_micro_tree.constructGraph(9, 1, 6, R)
macro_micro_tree.constructGraph(10, 9, 7, R)
macro_micro_tree.constructGraph(11, 10, 8, R)
macro_micro_tree.constructGraph(6, 10, 9, R)
macro_micro_tree.constructGraph(8, 10, 10, R)
macro_micro_tree.constructGraph(12, 8, 11, R)
macro_micro_tree.constructGraph(13, 12, 12, R)
macro_micro_tree.constructGraph(14, 11, 13, R)
macro_micro_tree.constructGraph(15, 1, 14, R)
macro_micro_tree.constructGraph(16, 15, 15, R)
macro_micro_tree.constructGraph(17, 15, 16, R)
macro_micro_tree.constructGraph(18, 17, 17, R)
macro_micro_tree.constructGraph(19, 18, 18, R)
macro_micro_tree.constructGraph(20, 13, 19, R)
macro_micro_tree.constructGraph(21, 13, 20, R)
macro_micro_tree.constructGraph(22, 2, 21, R)
macro_micro_tree.constructGraph(23, 2, 22, R)
macro_micro_tree.constructGraph(24, 3, 23, R)

macro_micro_tree.build_datastructure()
for i in range(1, 30):
    print("#########################")
    for j in range(macro_micro_tree.get_depth(i) + 1):
        anc = macro_micro_tree.level_ancestor(i, j)
        print("node {0}'s depth of {1} and its {2} ancstor is {3}".format(i, macro_micro_tree.get_depth(i), j, anc))

print(macro_micro_tree.level_ancestor(20, 8))
# print(macro_micro_tree.level_ancestor(6, 3))

