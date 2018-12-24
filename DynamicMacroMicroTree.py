import math
import collections
class DynamicMacroMicroTree:
    def __init__(self, R, n, maxlength):
        self.maxlength = maxlength
        self.edges = collections.defaultdict(list)
        self.jump = collections.defaultdict(list)
        self.depth = collections.defaultdict(int)
        self.maxdepth = math.log(maxlength, 2)
        self.size = collections.defaultdict(int)
        self.froms = collections.defaultdict(int)
        self.r0 = int(math.log(math.log(n, 2), 2))
        self.n = n
        self.M = int(math.log(n, 2)*0.5)
        self.ranks = collections.defaultdict(int)
        self.R = R
        self.microtree_root_table = collections.defaultdict(int)
        self.node_belongs_to = collections.defaultdict(int)
        self.tree_node_visit_sequence = []
        self.microset_size = collections.defaultdict(int)

    def constructGraph(self, c, p, R):
        self.edges[p].append(c)
        if c not in self.jump:
            self.jump[c].append(p)

            if c == R:
                self.depth[c] = 0
            if p == R:
                self.depth[c] = 1
        return
    def build_datastructure(self):
        """
        Collection of function to construct the whole data-structure
        :return:
        """
        self._dfs(self.R)
        self.d = collections.defaultdict(int)
        self.froms[self.R] = self.R
        self.jump, self.levelanc, self.macronodes,self.depth_macro_tree, self.macrotree_nodes, self.macrotree_parents = self._preprocess_macro_trees()
        self.jumpM, self.levelancM = self._process_jumpM()
        self.jump_micro = self._dfs_micro_tree()

    def rank(self, v, depth, size, maxdepth):
        """
        calculate the rank of each node in the tree.
        :param v:
        :param depth:
        :param size:
        :param maxdepth:
        :return:
        """
        if depth[v] == 0:
            return int(math.floor(math.log(self.n, 2)))

        idx = 0
        while idx < maxdepth:
            if depth[v] % int(math.pow(2, idx)) == 0 and size[v] >= math.pow(2, idx):
                idx += 1
            else:
                return idx

    def _dfs(self, v):
        """
        post-order visit of the tree
        :param v:
        :return:
        """
        children_visited = collections.defaultdict(int)
        stack = [v]
        self.depth[v] = 0
        self.size[v] = 1
        while stack:
            v = stack[-1]
            if v in self.edges:
                if children_visited[v] < len(self.edges[v]):
                    self.size[self.edges[v][children_visited[v]]] = 1
                    self.froms[self.edges[v][children_visited[v]]] = v
                    self.depth[self.edges[v][children_visited[v]]] = self.depth[v] + 1
                    stack.append(self.edges[v][children_visited[v]])
                    children_visited[v] += 1
                else:
                    item_pop = stack.pop()
                    self.tree_node_visit_sequence.append(item_pop)
                    self.size[self.froms[item_pop]] += self.size[item_pop]
            else:
                item_pop = stack.pop()
                self.tree_node_visit_sequence.append(item_pop)
                self.size[self.froms[item_pop]] += self.size[item_pop]


    def _construct_macro_tree(self, jump, depth, edges, c, p, R):
        edges[p].append(c)
        if c not in jump:
            jump[c].append(p)
            if R == p:
                depth[c] = 1
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
        jump = collections.defaultdict(list)
        depth = collections.defaultdict(int)
        edges = collections.defaultdict(list)
        parents = collections.defaultdict(int)
        size = collections.defaultdict(int)
        macronodes = collections.defaultdict(int)
        node_in_macrotrees = collections.defaultdict(int)

        ### Step 1: calculate ranks for each node
        for i in range(1, self.n):
            self.ranks[i] = self.rank(i, self.depth, self.size, self.maxdepth)
            macronodes[i] = 1 if self.ranks[i] >= self.r0 else -1


        ### Step 2: retrieve macro node
        stack = [self.R]
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
                    self._construct_macro_tree(jump, depth, edges, cur_node, v, self.R)
                    if cur_node not in node_in_macrotrees:
                        node_in_macrotrees[cur_node] = 1
                    if v not in node_in_macrotrees:
                        node_in_macrotrees[v] = 1

            if cur_node in self.edges:
                stack = self.edges[cur_node] + stack

        ### Step 3: traverse the macro tree and calculate depth
        stack = [R]
        while stack:
            v = stack.pop(0)
            if v in edges:
                for w in edges[v]:
                    depth[w] = depth[v] + 1
                    parents[w] = v
                stack = edges[v] + stack

        ### Step 4: calculate ranks for each macro node
        ranks_macro_node = collections.defaultdict(int)
        for i in range(1, self.n):
            ranks_macro_node[i] = self.rank(i, depth, size, self.maxdepth)

        ### Step 5: calculate levelanc for macro trees
        levelanc = collections.defaultdict(list)
        for v in jump.keys():
            if v in parents:
                tmp_node = v
                for j in range(depth[v]):
                    levelanc[tmp_node].append(parents[tmp_node])
                    tmp_node = parents[tmp_node]
        levelanc[self.R].append(self.R)
        ### Step 6: build jump table for macro trees
        jump = collections.defaultdict(list)
        stack = [R]
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
        return jump, levelanc, macronodes, depth, node_in_macrotrees, parents

    def _process_jumpM(self):
        jumpM = [-1] * self.n
        levelancM = collections.defaultdict(list)
        stack = [self.R]
        jumpM[self.R] = self.R
        while stack:
            v = stack.pop(0)
            cur, idx = v, 1
            levelancM[cur] = [cur]
            # assign M ancestor to each node
            if self.depth[self.froms[v]] % self.M == 0:
                jumpM[cur] = self.froms[v]
            else:
                jumpM[cur] = jumpM[self.froms[v]]
            v = cur
            ## add ancestors to each levelancM
            while v != self.R and len(levelancM[cur]) <= self.M:
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

        ## deal with unprocessed nodes
        for v, edges in self.edges.items():
            if len(edges) > 0 and v != 0:
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
                jump_table.extend(jump[jump_table[-1]][1:self.M + 1 - len(jump_table) + 1])

        return jump

    def _level_ancestor(self, v, x):
        i = k = int(math.log(x +1, 2))
        d = self.depth_macro_tree[v] - x
        prev = v
        while 2 * (self.depth_macro_tree[v] - d) >= int(math.pow(2, k)):
            if 0 < len(self.jump[v]) < i:
              i -= 1
            elif len(self.jump[v]) == 0:
                break

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
    def _add_leaf(self, v, p):
        new_microset = False
        if self.microset_size[self.node_belongs_to[p]] >= self.M:
            self.node_belongs_to[v] = self.microset_idx
            self.microtree_root_table[self.microset_idx] = v
            new_microset = True
            self.microset_idx += 1
        else:
            self.node_belongs_to[v] = self.node_belongs_to[p]
        self.microset_size[self.node_belongs_to[v]] = 1
        self.depth[v] = self.depth[p] + 1
        self.jump_micro[v] = [v, p] + self.jump_micro[p][1:self.M+1]
        return new_microset, self.microset_idx
    def _add_leaf_step(self, v, p):
        self.levelanc[v] = [v]
        self.macrotree_parents[v] = p
        self.macronodes[v] = 1
        self.depth_macro_tree[v] = self.depth_macro_tree[p] + 1
        self.macrotree_nodes[v] = 1
        for i in range(int(math.log(self.depth_macro_tree[v], 2))):
            w = p
            if self.depth_macro_tree[p] % int(math.pow(2, i)) == 0:
                w = p
            else:
                if i >= len(self.jump[self.macrotree_parents[v]]):
                    w = self.jump[self.macrotree_parents[v]][-1]
                else:
                    w = self.jump[self.macrotree_parents[v]][i]
            self.jump[v].append(w)
            lastanc2_w = self.levelanc[w][-1]
            if self.macrotree_parents[lastanc2_w] != -1:
                self.levelanc[w].append(self.macrotree_parents[lastanc2_w])


    def add_leaf(self, v, p):
        self.froms[v] = p
        new_microset, microset_id = self._add_leaf(v, p)
        if self.depth[p] % self.r0 == 0:
            self.jumpM[v] = p
        else:
            self.jumpM[v] = self.jumpM[p]

        if new_microset:
            self.levelancM[v] = [v, p] + self.levelancM[p][1: self.M]
        else:
            if len(self.levelancM[self.microtree_root_table[self.node_belongs_to[p]]]) < self.M:
                last_anc = self.levelancM[self.microtree_root_table[self.node_belongs_to[p]]][-1]
                self.levelancM[self.microtree_root_table[self.node_belongs_to[p]]].append(self.froms[last_anc])

        if self.jumpM[v] not in self.macrotree_nodes:
            self._add_leaf_step(self.jumpM[v], self.jumpM[self.jumpM[v]])

n, R = 400, 2
macro_micro_tree = DynamicMacroMicroTree(R, n+1, 3000)

macro_micro_tree.constructGraph(2, 0, R)
macro_micro_tree.constructGraph(5, 2, R)
macro_micro_tree.constructGraph(3, 5, R)
macro_micro_tree.constructGraph(4, 5, R)
macro_micro_tree.constructGraph(1, 5, R)
macro_micro_tree.constructGraph(7, 1, R)
macro_micro_tree.constructGraph(9, 1, R)
macro_micro_tree.constructGraph(10, 9, R)
macro_micro_tree.constructGraph(11, 10, R)
macro_micro_tree.constructGraph(6, 10, R)
macro_micro_tree.constructGraph(8, 10, R)
macro_micro_tree.constructGraph(12, 8,  R)
macro_micro_tree.constructGraph(13, 12, R)
macro_micro_tree.constructGraph(14, 11, R)
macro_micro_tree.constructGraph(15, 1,  R)
macro_micro_tree.constructGraph(16, 15, R)
macro_micro_tree.constructGraph(17, 15, R)
macro_micro_tree.constructGraph(18, 17, R)
macro_micro_tree.constructGraph(19, 18, R)
macro_micro_tree.constructGraph(20, 13, R)
macro_micro_tree.constructGraph(21, 13, R)
macro_micro_tree.constructGraph(22, 2, R)
macro_micro_tree.constructGraph(23, 2, R)
macro_micro_tree.constructGraph(24, 3, R)

import random
macro_micro_tree.build_datastructure()
for i in range(24, n):
    p = random.randint(1, i-1)
    macro_micro_tree.add_leaf(i, p)
results = collections.defaultdict(list)
for i in range(1, n):
    print("#########################")
    for j in range(macro_micro_tree.get_depth(i) + 1):
        anc = macro_micro_tree.level_ancestor(i, j)
        results[i].append(anc)
        print("node {0}'s depth of {1} and its {2} ancstor is {3}".format(i, macro_micro_tree.get_depth(i), j, anc))
