'''
MIT License

Copyright (c) 2021 Intel Labs

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
from enum import Enum


class NodeType(Enum):
    Internal = 0,
    NumLit = 1,
    CharLit = 2,
    StringLit = 3,
    GlobalVar = 4,
    GlobalFun = 5,
    LocalVar = 6,
    LocalFun = 7,
    Error = 8


class CassNode:
    def __init__(self, node_type, label='', children=[]):
        self.node_type = node_type
        self.children = children
        self.prev_use = None
        self.next_use = None
        self.parent = None
        self.child_id = 0

        self.label = label

        if len(label) == 0:
            self.n = label

        elif node_type == NodeType.Internal:
            self.n = 'N' + label

        # elif node_type == NodeType.LocalVar or node_type == NodeType.LocalFun:
        #     self.n = 'T$VAR'

        else:
            self.n = 'T' + label

        self.features = []


class CassTree:
    def __init__(self, nodes, leaf_nodes):
        self.nodes = nodes
        self.leaf_nodes = leaf_nodes
        self.root = nodes[0]
        self.leaf_ranges = self._compute_leaf_ranges()

    def _compute_leaf_ranges(self):
        node2leaf_id = {}
        leaf_ranges = {}
        for i, node in enumerate(self.leaf_nodes):
            node2leaf_id[node] = i

        def compute_leaf_ranges_rec(node):
            if len(node.children) == 0:
                if node.node_type != NodeType.Internal:
                    x = node2leaf_id[node]
                    leaf_ranges[node] = (x, x + 1)
                else:
                    leaf_ranges[node] = (-1, -1)
            else:
                for c in node.children:
                    compute_leaf_ranges_rec(c)
                l = -1
                for c in node.children:
                    x = leaf_ranges[c][0]
                    if x >= 0:
                        l = x
                        break
                r = -1
                if l >= 0:
                    for c in reversed(node.children):
                        x = leaf_ranges[c][1]
                        if x >= 0:
                            r = x
                            break
                leaf_ranges[node] = (l, r)

        compute_leaf_ranges_rec(self.root)
        return leaf_ranges

    def _get_context(self, node):
        p = node.parent
        if p is None:
            return None
        if p.label != 'field_expression':
            return (node.child_id, p.n)
        else:
            for i in range(*(self.leaf_ranges[p])):
                l = self.leaf_nodes[i]
                if l.node_type == NodeType.GlobalVar or l.node_type == NodeType.GlobalFun:
                    return l.n
            return None

    def featurize(self):
        for i, node in enumerate(self.leaf_nodes):
            node.features.append(node.n)

            p = node
            for _ in range(3):
                cid = p.child_id
                p = p.parent
                if p is None:
                    break
                node.features.append((node.n, cid, p.n))

            if i > 0:
                sib = self.leaf_nodes[i - 1]
                node.features.append((sib.n, node.n))
            if i < len(self.leaf_nodes) - 1:
                sib = self.leaf_nodes[i + 1]
                node.features.append((node.n, sib.n))

            if node.prev_use is not None:
                prev_ctx = self._get_context(node.prev_use)
                ctx = self._get_context(node)
                if prev_ctx is not None and ctx is not None:
                    node.features.append((prev_ctx, ctx))
            if node.next_use is not None:
                ctx = self._get_context(node)
                next_ctx = self._get_context(node.next_use)
                if ctx is not None and next_ctx is not None:
                    node.features.append((ctx, next_ctx))

        features = []
        for n in self.leaf_nodes:
            features += n.features

        return features


def load_file(file_name):
    casses = []
    with open(file_name) as f:
        for line in f:
            cass = deserialize(line)
            if cass is not None:
                casses.append(cass)
    return casses


def deserialize(s):
    tokens = s.strip().split('\t')
    num_tokens = len(tokens)
    if num_tokens == 0:
        return None

    num_nodes = int(tokens[0])

    nodes = []
    leaf_nodes = []

    i = 1

    while i < num_tokens:
        node_type_label = tokens[i]
        i += 1
        node_type_str = node_type_label[0]
        label = node_type_label[1:]
        if node_type_str == 'I':
            num_child = int(tokens[i])
            i += 1
            nodes.append(
                CassNode(NodeType.Internal, label, [None] * num_child))
        elif node_type_str == 'N':
            node = CassNode(NodeType.NumLit, label)
            nodes.append(node)
            leaf_nodes.append(node)
        elif node_type_str == 'C':
            node = CassNode(NodeType.CharLit, label)
            nodes.append(node)
            leaf_nodes.append(node)
        elif node_type_str == 'S':
            node = CassNode(NodeType.StringLit, label)
            nodes.append(node)
            leaf_nodes.append(node)
        elif node_type_str == 'V':
            node = CassNode(NodeType.GlobalVar, label)
            nodes.append(node)
            leaf_nodes.append(node)
        elif node_type_str == 'F':
            node = CassNode(NodeType.GlobalFun, label)
            nodes.append(node)
            leaf_nodes.append(node)
        elif node_type_str == 'v':
            prev_use = int(tokens[i])
            next_use = int(tokens[i + 1])
            i += 2
            node = CassNode(NodeType.LocalVar, label)
            node.prev_use = prev_use
            node.next_use = next_use
            nodes.append(node)
            leaf_nodes.append(node)
        elif node_type_str == 'f':
            prev_use = int(tokens[i])
            next_use = int(tokens[i + 1])
            i += 2
            node = CassNode(NodeType.LocalFun, label)
            node.prev_use = prev_use
            node.next_use = next_use
            nodes.append(node)
            leaf_nodes.append(node)
        elif node_type_str == 'E':
            node = CassNode(NodeType.Error)
            nodes.append(node)
            leaf_nodes.append(node)
        else:
            raise Exception()

    assert num_nodes == len(nodes)

    for n in nodes:
        if n.node_type == NodeType.LocalVar or n.node_type == NodeType.LocalFun:
            if n.prev_use >= 0:
                n.prev_use = nodes[n.prev_use]
            else:
                n.prev_use = None
            if n.next_use >= 0:
                n.next_use = nodes[n.next_use]
            else:
                n.next_use = None

    root, rem_nodes = build_tree_rec(nodes)

    assert root == nodes[0]
    assert len(rem_nodes) == 0

    return CassTree(nodes, leaf_nodes)


def build_tree_rec(nodes):
    node = nodes[0]
    nodes = nodes[1:]
    for i in range(len(node.children)):
        child, nodes = build_tree_rec(nodes)
        child.parent = node
        child.child_id = i
        node.children[i] = child
    return node, nodes
