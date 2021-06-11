import argparse
import multiprocessing
import os
import tqdm
import subprocess
from tree_sitter import Language, Parser
from pathlib import Path


tree_sitter_lib_path = str(Path(__file__).absolute().parent /
                           'build' / 'tree-sitter-languages.so')
Language.build_library(
    tree_sitter_lib_path,
    [
        Path(__file__).absolute().parent.parent /
        'cass-extractor' / 'tree-sitter' / 'tree-sitter-c',
        Path(__file__).absolute().parent.parent /
        'cass-extractor' / 'tree-sitter' / 'tree-sitter-cpp'
    ]
)

header = b'''
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <iostream>
using namespace std;
'''
flags = '-Wno-everything -S -emit-llvm -std=c++11 -DSIZE=64 -DSIZEE=64 -DROW=64 -DCOL=64 -DN=64 -DM=64 -DLEN=64'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clangxx', '-c', type=str, required=True)
    parser.add_argument('--dataset', '-d', type=str,
                        choices=('poj', 'gcj'), required=True)
    parser.add_argument('--input-dir', '-i', type=str, required=True)
    parser.add_argument('--output-dir', '-o', type=str, required=True)
    parser.add_argument('--filter-list', '-f', type=str, default=None)
    parser.add_argument('--num-workers', '-p', type=int,
                        default=os.cpu_count())
    args = parser.parse_args()
    return args


def filter_program(arguments):
    problem, solution, args = arguments

    with open(os.path.join(args.input_dir, problem, solution), 'rb') as f:
        src_bytes = f.read()

    parser = Parser()
    language = 'c' if args.dataset == 'poj' else 'cpp'
    parser.set_language(Language(tree_sitter_lib_path, language))
    if parser.parse(src_bytes).root_node.has_error:
        return problem, solution

    if args.dataset == 'poj':
        patched_src = header + src_bytes.replace(b'void main', b'int main')
    else:
        patched_src = src_bytes
    cmd = f'{args.clangxx} -x c++ {flags} - -o /dev/null'
    try:
        subprocess.run(cmd.split(' '), check=True, input=patched_src,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return problem, solution

    os.makedirs(os.path.join(args.output_dir, problem), exist_ok=True)
    with open(os.path.join(args.output_dir, problem, solution), 'wb') as f:
        f.write(src_bytes)

    return None


def main():
    args = parse_args()

    filter_set = set()
    if args.filter_list is not None:
        with open(args.filter_list) as f:
            for line in f:
                p, s = line.strip().split('/')
                filter_set.add((p, s))

    solution_list = []
    for problem in os.listdir(args.input_dir):
        problem_dir = os.path.join(args.input_dir, problem)
        for solution in os.listdir(problem_dir):
            if (problem, solution) in filter_set:
                continue
            solution_path = os.path.join(problem_dir, solution)
            solution_list.append((problem, solution))

    def gen_args():
        for problem, solution in solution_list:
            yield problem, solution, args

    err_solutions = []
    with multiprocessing.Pool(args.num_workers) as pool:
        for r in tqdm.tqdm(
            pool.imap_unordered(
                filter_program,
                gen_args()
            ),
            total=len(solution_list)
        ):
            if r is None:
                continue
            err_solutions.append(r)

    total = len(solution_list)
    remain = total - len(err_solutions)
    print(f'{remain}/{total}')


if __name__ == "__main__":
    main()
