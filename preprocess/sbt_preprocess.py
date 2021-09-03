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
import argparse
import os
import pickle
from collections import defaultdict
import numpy as np
import multiprocessing
import pickle

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from cass.cass import load_file


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cass-dir', '-i', type=str, required=True)
    parser.add_argument('--output-dir', '-o', type=str, required=True)
    parser.add_argument('--split', '-s', type=str, required=True)
    parser.add_argument('--min-freq', '-t', type=int, default=5)
    parser.add_argument('--num-workers', '-p', type=int,
                        default=os.cpu_count())
    return parser.parse_args()


def sbt(root, seq):
    seq.append('(')
    seq.append(root.n)
    for c in root.children:
        sbt(c, seq)
    seq.append(')')
    seq.append(root.n)


def load_sbt(arguments):
    pid, problem_dir, solution = arguments
    casss = load_file(os.path.join(problem_dir, solution))
    assert len(casss) > 0
    seqs = []
    for cass in casss:
        seq = []
        sbt(cass.root, seq)
        if cass.fun_sig_node.n is not None:
            seq = ['{', cass.fun_sig_node.n] + seq + ['}', cass.fun_sig_node.n]
        seqs.append(seq)
    return pid, solution, seqs


def load_sbts(args):
    cass_dir = args.cass_dir

    paths = []
    for problem in os.listdir(cass_dir):
        problem_dir = os.path.join(cass_dir, problem)
        if os.path.isdir(problem_dir):
            pid = problem
            for solution in os.listdir(problem_dir):
                if solution.endswith('.cas'):
                    paths.append(
                        (pid, problem_dir, solution))

    all_seqs = defaultdict(dict)
    with multiprocessing.Pool(args.num_workers) as pool:
        for pid, solution, seqs in pool.imap_unordered(load_sbt, paths, chunksize=10):
            if seqs is not None:
                all_seqs[pid][solution[:-4]] = seqs

    return dict(all_seqs)


def get_freqs(arguments):
    problem_seqs, training_solutions = arguments
    freqs = defaultdict(int)
    for train_sol in training_solutions:
        solution_seqs = problem_seqs[train_sol]
        for seq in solution_seqs:
            for t in seq:
                freqs[t] += 1
    return freqs


def build_vocab(all_seqs, split, args):
    training_set = split[0]

    def gen_args():
        for pid, training_solutions in training_set.items():
            yield all_seqs[pid], training_solutions

    freqs = defaultdict(int)
    with multiprocessing.Pool(args.num_workers) as pool:
        for freqs_local in pool.imap_unordered(get_freqs, gen_args(), chunksize=10):
            for k, v in freqs_local.items():
                freqs[k] += v

    vocab_count_list = sorted(
        freqs.items(), key=lambda kv: kv[1], reverse=True)
    total = sum(map(lambda wc: wc[1], vocab_count_list))
    in_vocab = 0
    vocab = {}
    for i, (word, count) in enumerate(vocab_count_list):
        if count // 2 < args.min_freq:  # Every word is counted twice in SBT
            break
        vocab[word] = i + 1  # Reserve 0 for UNK
        in_vocab += count

    vocab[''] = 0

    print(f'Vocab size: {len(vocab)}/{len(vocab_count_list)}')
    print(f'Vocab coverage: {in_vocab}/{total} = {in_vocab/total:.2%}')

    return vocab


def preprocess(arguments):
    pid, problem_seqs, vocab = arguments
    data = {}
    for solution, solution_seqs in problem_seqs.items():
        solution_data = []
        for seq in solution_seqs:
            solution_data.append(
                np.asarray([vocab.get(t, 0) for t in seq], dtype=np.int32))
        data[solution] = solution_data
    return pid, data


def preprocess_dataset(all_seqs, vocab, args):
    dataset = {}
    def gen_args():
        for i, s in all_seqs.items():
            yield i, s, vocab
    with multiprocessing.Pool(args.num_workers) as pool:
        for pid, data in pool.imap_unordered(preprocess, gen_args()):
            dataset[pid] = data
    return dataset


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.split, 'rb') as f:
        split = pickle.load(f)

    all_seqs = load_sbts(args)

    vocab = build_vocab(all_seqs, split, args)

    with open(os.path.join(args.output_dir, 'vocab.pkl'), 'wb') as f:
        pickle.dump(vocab, f)

    dataset = preprocess_dataset(all_seqs, vocab, args)

    with open(os.path.join(args.output_dir, 'dataset.pkl'), 'wb') as f:
        pickle.dump(dataset, f)


if __name__ == "__main__":
    main()
