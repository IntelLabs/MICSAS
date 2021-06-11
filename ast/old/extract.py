import os
import argparse
import multiprocessing
import tqdm
from tree_sitter import Language, Parser


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', '-i', type=str, required=True)
    parser.add_argument('--output-dir', '-o', type=str, required=True)
    parser.add_argument('--num-workers', '-p', type=int, default=None)
    parser.add_argument('--dataset', '-d', type=str, choices=('poj', 'gcj'),
                        required=True)
    return parser.parse_args()


class ASTNode:
    types_with_operator = {
        'binary_expression',
        'unary_expression',
        'update_expression',
        'assignment_expression'
    }

    def __init__(self, ts_node):
        self.children = []
        self.parent = None
        self.child_id = None
        self.is_leaf = None
        self.leaf_num = 0

        node_type = ts_node.type
        self.raw_type = node_type
        if node_type in self.types_with_operator:
            node_type += self.get_operator(ts_node)
        self.node_type = node_type

        self.start_byte = ts_node.start_byte
        self.end_byte = ts_node.end_byte

    @staticmethod
    def get_operator(ts_node):
        for n in ts_node.children:
            for c in n.type:
                if not c.isalnum() and c != '_':
                    return n.type
        return ''


class AST:
    def __init__(self, ts_root, src_bytes):
        self.src_bytes = src_bytes

        nodes = []

        # DFS tree clone
        stack = [(ts_root, None, None)]
        while len(stack) > 0:
            ts_node, remaining_children, parent_or_node = stack.pop()
            if remaining_children is None:
                parent = parent_or_node
                ast_node = ASTNode(ts_node)
                ast_node.parent = parent
                if parent is not None:
                    ast_node.child_id = len(parent.children)
                    parent.children.append(ast_node)
                nodes.append(ast_node)
                if ts_node.type.endswith('literal'):
                    children = []
                else:
                    children = self.get_named_children(ts_node)
                stack.append((ts_node, children, ast_node))
                if len(children) > 0:
                    child = children[0]
                    stack.append((child, None, ast_node))
            elif len(remaining_children) > 0:
                ast_node = parent_or_node
                stack.append((ts_node, remaining_children[1:], ast_node))
                if len(remaining_children) > 1:
                    child = remaining_children[1]
                    stack.append((child, None, ast_node))

        for node in reversed(nodes):
            node.is_leaf = len(node.children) == 0
            if node.is_leaf:
                node.leaf_num = 1
            if node.parent is not None:
                node.parent.leaf_num += node.leaf_num

        self.nodes = nodes

    @staticmethod
    def get_named_children(ts_node):
        return [child for child in ts_node.children if child.is_named and child.type != 'comment']

    def __str__(self):
        strs = []
        def str_rec(node, depth):
            strs.append('  ' * depth)
            strs.append(node.node_type)
            if node.is_leaf:
                strs.append(':\t')
                strs.append(self.src_bytes[node.start_byte:node.end_byte].decode())
            strs.append('\n')
            for child in node.children:
                str_rec(child, depth + 1)

        str_rec(self.nodes[0], 0)

        return ''.join(strs)

    def serialize(self, out):
        out.write(str(len(self.nodes)))
        for node in self.nodes:
            t = node.node_type
            out.write('\t')
            out.write(t)
            out.write('\t')
            if t.endswith('literal') or (t.endswith('identifier') and not t.startswith('scoped')):
                assert len(node.children) == 0
                text = self.src_bytes[node.start_byte:node.end_byte].decode()
                text = ''.join(c for c in text if c not in {'\t', '\n', '\r'})
                out.write(text)
            else:
                out.write(str(len(node.children)))


def extract_from_solution(arguments):
    solution_path, output_path, args = arguments

    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    parser = Parser()
    parser.set_language(Language(args.ts_lib_path, args.language))

    with open(solution_path, 'rb') as f:
        src_bytes = f.read()

    tree = parser.parse(src_bytes)
    ast = AST(tree.root_node, src_bytes)

    with open(output_path, 'w') as f:
        ast.serialize(f)


def main():
    args = parse_args()

    language = 'c' if args.dataset == 'poj' else 'cpp'
    this_dir = os.path.dirname(os.path.abspath(__file__))
    ts_lib_path = os.path.join(this_dir, 'build/tree-sitter-languages.so')
    ts_repo = os.path.abspath(os.path.join(
        this_dir,
        f'../cass-extractor/tree-sitter/tree-sitter-{language}'
    ))
    Language.build_library(ts_lib_path, [ts_repo])
    args.ts_lib_path = ts_lib_path
    args.language = language

    problem_list = []
    for problem in os.listdir(args.input_dir):
        problem_dir = os.path.join(args.input_dir, problem)
        for solution in os.listdir(problem_dir):
            solution_path = os.path.join(problem_dir, solution)
            output_path = os.path.join(args.output_dir, problem, solution[:-3] + 'ast')
            problem_list.append((solution_path, output_path, args))

    with multiprocessing.Pool(args.num_workers) as pool:
        for _ in tqdm.tqdm(
            pool.imap_unordered(
                extract_from_solution,
                problem_list),
                total=len(problem_list)):
            pass


if __name__ == "__main__":
    main()
