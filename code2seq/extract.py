import os
import argparse
import regex
import multiprocessing
import tqdm
from tree_sitter import Language, Parser


internal_separator = '$'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', '-i', type=str, required=True)
    parser.add_argument('--output-dir', '-o', type=str, required=True)
    parser.add_argument('--num-workers', '-p', type=int, default=None)
    parser.add_argument('--dataset', '-d', type=str, choices=('poj', 'gcj'),
                        required=True)

    parser.add_argument('--max-path-length', type=int, default=8)
    parser.add_argument('--max-path-width', type=int, default=2)
    parser.add_argument('--max-child-id', type=int, default=2147483647)
    parser.add_argument('--max-leaf-num', type=int, default=5000)
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


def normalize_name(s):
    s = s.lower()
    s = regex.sub('\\s+', '', s)
    return s


def split_to_subtokens(s):
    s = s.strip()
    ts = regex.split("(?<=[a-z])(?=[A-Z])|_|$|[0-9]|(?<=[A-Z])(?=[A-Z][a-z])|\\s+",
                     s, flags=regex.V1)
    return list(filter(lambda x: len(x) > 0, map(normalize_name, ts)))


def collect_functions(ast, args):
    functions = []

    for node in ast.nodes:
        if node.node_type != 'function_definition':
            continue

        if node.leaf_num > args.max_leaf_num:
            continue

        function_dict = {child.node_type: child for child in node.children}

        # Check if body exists
        body = function_dict.get('compound_statement', None)
        if body is None or len(body.children) == 0:
            continue

        # Get function name
        declarator = function_dict.get('function_declarator', None)
        if declarator is None:
            continue
        declarator_dict = {child.node_type: child for child in declarator.children}
        for name_field in ('identifier', 'field_identifier', 'operator_name'):
            function_name = declarator_dict.get(name_field, None)
            if function_name is not None:
                break
        if function_name is None:
            continue

        # Avoid including the function name in path features
        function_name.leaf_name = 'FUNCTION_NAME'

        # Normalize function name
        function_name = ast.src_bytes[function_name.start_byte:function_name.end_byte].decode()
        split_name = internal_separator.join(split_to_subtokens(function_name))
        if len(split_name) == 0:
            split_name = normalize_name(function_name)
        if len(split_name) == 0:
            continue

        node.function_name = split_name
        functions.append(node)

    return functions


parent_types_to_add_child_id = {
    'assignment_expression',
    'subscript_expression',
    'field_expression',
    'call_expression'
}


def get_tree_stack(node, lca):
    stack = []
    while node is not lca:
        stack.append(node)
        node = node.parent
    return stack


def generate_path(source, target, lca, args):
    source_stack = get_tree_stack(source, lca)
    target_stack = get_tree_stack(target, lca)

    i = len(source_stack) - 1
    j = len(target_stack) - 1

    path_length = i + j + 2
    if path_length > args.max_path_length:
        return ''

    if i >= 0 and j >= 0:
        path_width = target_stack[j].child_id - source_stack[i].child_id
        if path_width > args.max_path_width:
            return ''

    path = []

    for k in range(0, i + 1):
        node = source_stack[k]
        child_id = ''
        if k == 0 or node.parent.raw_type in parent_types_to_add_child_id:
            child_id = str(min(args.max_child_id, node.child_id))
        path.extend((node.node_type, child_id, internal_separator))

    child_id = ''
    if lca.parent.raw_type in parent_types_to_add_child_id:
        child_id = str(min(args.max_child_id, lca.child_id))
    path.extend((lca.node_type, child_id))

    for k in range(j, -1, -1):
        node = target_stack[k]
        child_id = ''
        if k == 0 or node.parent.raw_type in parent_types_to_add_child_id:
            child_id = str(min(args.max_child_id, node.child_id))
        path.extend((internal_separator, node.node_type, child_id))

    return ''.join(path)


numbers_to_keep = {
    '0', '1', '32', '64'
}


def process_leaf(node, ast):
    if not hasattr(node, 'leaf_name'):
        name = ast.src_bytes[node.start_byte:node.end_byte].decode()
        if node.node_type == 'number_literal':
            if name in numbers_to_keep:
                leaf_name = name
            else:
                leaf_name = 'NUM'
        elif node.node_type == 'string_literal':
            leaf_name = 'STR'
        elif node.node_type == 'char_literal':
            leaf_name = 'CHR'
        elif node.node_type.endswith('identifier') or node.node_type.endswith('type'):
            split_name = internal_separator.join(split_to_subtokens(name))
            if len(split_name) == 0:
                split_name = normalize_name(name)[:50]
                if len(split_name) == 0:
                    split_name = 'BLANK'
            leaf_name = split_name
        else:
            leaf_name = node.node_type.upper()
        node.leaf_name = leaf_name


def find_ancester(node):
    stack = [node]
    while len(stack) > 0:
        n = stack[len(stack) - 1]
        na = n.ancestor
        if n is na:
            break
        stack.append(na)
    for n in stack:
        n.ancestor = na
    return na


def generate_path_features_for_function(function, ast, args):
    leaves = []
    path_features = []

    function.ancestor = function
    stack = [(function, None)]
    while len(stack) > 0:
        node, remaining_children = stack.pop()
        if remaining_children is None:
            remaining_children = node.children
            stack.append((node, remaining_children))
            if len(remaining_children) > 0:
                child = remaining_children[0]
                child.ancestor = child
                stack.append((child, None))
        elif len(remaining_children) > 0:
            child = remaining_children[0]
            child.ancestor = node
            stack.append((node, remaining_children[1:]))
            if len(remaining_children) > 1:
                child = remaining_children[1]
                child.ancestor = child
                stack.append((child, None))
        else:
            if node.is_leaf:
                process_leaf(node, ast)
                for prev_leaf in leaves:
                    lca = find_ancester(prev_leaf)
                    path = generate_path(prev_leaf, node, lca, args)
                    if len(path) > 0:
                        path_features.append((
                            prev_leaf.leaf_name,
                            path,
                            node.leaf_name
                        ))
                leaves.append(node)

    return path_features


def extract_features(ast, args):
    functions = collect_functions(ast, args)

    functions_features = []
    for function in functions:
        path_features = generate_path_features_for_function(function, ast, args)
        if len(path_features) > 0:
            functions_features.append((function.function_name, path_features))

    return functions_features


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

    functions_features = extract_features(ast, args)

    with open(output_path, 'w') as f:
        for fn, fs in functions_features:
            f.write(fn)
            for ll, path, rl in fs:
                f.write(f' {ll},{path},{rl}')
            f.write('\n')


def main():
    args = parse_args()

    language = 'c' if args.dataset == 'poj' else 'cpp'
    this_dir = os.path.dirname(os.path.abspath(__file__))
    ts_lib_path = os.path.join(this_dir, f'build/tree-sitter-{language}.so')
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
            output_path = os.path.join(args.output_dir, problem, solution[:-3] + 'c2s')
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
