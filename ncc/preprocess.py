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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ncc-dir', '-i', type=str, required=True)
    parser.add_argument('--output-dir', '-o', type=str, required=True)
    parser.add_argument('--vocab-dir', '-v', type=str,
                        default=os.path.join(
                            os.path.dirname(os.path.abspath(__file__)),
                            'ncc/published_results/vocabulary'))
    parser.add_argument('--split', '-s', type=str, required=True)
    parser.add_argument('--num-workers', '-p', type=int,
                        default=os.cpu_count())
    return parser.parse_args()


def load_seq(arguments):
    pid, problem_dir, solution = arguments
    with open(os.path.join(problem_dir, solution), 'r') as f:
        seq = np.asarray(list(map(int, f.read().splitlines())), dtype=np.int32)
    return pid, solution, seq


def load_seqs(args):
    ncc_dir = args.ncc_dir

    paths = []
    for problem in os.listdir(ncc_dir):
        if not problem.startswith('seq'):
            continue
        problem_dir = os.path.join(ncc_dir, problem)
        pid = os.path.split(problem_dir)[1].split('_')[1]
        for solution in os.listdir(problem_dir):
            if solution.endswith('.csv'):
                paths.append((pid, problem_dir, solution))

    all_seqs = defaultdict(dict)
    with multiprocessing.Pool(args.num_workers) as pool:
        for pid, solution, seq in pool.imap_unordered(load_seq, paths, chunksize=10):
            if seq is not None:
                sol = solution[:-4].split('_')[0]
                all_seqs[pid][sol] = seq

    return dict(all_seqs)


def load_vocab(args):
    dictionary_pickle = os.path.join(args.vocab_dir, 'dic_pickle')
    with open(dictionary_pickle, 'rb') as f:
        vocab = pickle.load(f)
    return vocab


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.split, 'rb') as f:
        split = pickle.load(f)

    dataset = load_seqs(args)

    vocab = load_vocab(args)

    with open(os.path.join(args.output_dir, 'vocab.pkl'), 'wb') as f:
        pickle.dump(vocab, f)

    with open(os.path.join(args.output_dir, 'dataset.pkl'), 'wb') as f:
        pickle.dump(dataset, f)


if __name__ == "__main__":
    main()
