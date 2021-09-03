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


internal_separator = '$'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--c2s-dir', '-i', type=str, required=True)
    parser.add_argument('--output-dir', '-o', type=str, required=True)
    parser.add_argument('--split', '-s', type=str, required=True)

    parser.add_argument('--subtoken-vocab-size', '-svs', type=int, default=186277)
    parser.add_argument('--max-contexts', '-mc', type=int, default=200)
    parser.add_argument('--max-data-contexts', '-mdc', type=int, default=1000)

    parser.add_argument('--max-path-length', type=int, default=8)

    parser.add_argument('--num-workers', '-p', type=int,
                        default=os.cpu_count())
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def load_c2s(filename):
    c2s = []
    with open(filename) as f:
        for line in f:
            line = line.strip().split(' ')
            for s in line[1:]:
                ll, path, rl = s.split(',')
                c2s.append((ll.split(internal_separator), path.split(internal_separator), rl.split(internal_separator)))
    return c2s


def get_freqs(arguments):
    pid, training_solutions, args = arguments
    subtoken_freqs = defaultdict(int)
    node_freqs = defaultdict(int)
    for train_sol in training_solutions:
        c2s = load_c2s(os.path.join(args.c2s_dir, pid, train_sol + '.c2s'))
        for ll, path, rl in c2s:
            for t in ll:
                subtoken_freqs[t] += 1
            for t in rl:
                subtoken_freqs[t] += 1
            for n in path:
                node_freqs[n] += 1
    return subtoken_freqs, node_freqs


def build_vocab(split, args):
    training_set = split[0]

    def gen_args():
        for pid, training_solutions in training_set.items():
            yield pid, training_solutions, args

    subtoken_freqs = defaultdict(int)
    node_freqs = defaultdict(int)
    with multiprocessing.Pool(args.num_workers) as pool:
        for subtoken_freqs_local, node_freqs_local in pool.imap_unordered(
            get_freqs, gen_args(), chunksize=10
        ):
            for k, v in subtoken_freqs_local.items():
                subtoken_freqs[k] += v
            for k, v in node_freqs_local.items():
                node_freqs[k] += v

    subtoken_vocab = {'': 0}  # UNK
    for i, (st, _) in enumerate(sorted(subtoken_freqs.items(), key=lambda kv: kv[1], reverse=True)[:args.subtoken_vocab_size]):
        subtoken_vocab[st] = i + 1
    
    node_vocab = {'': 0}  # UNK
    for i, (n, _) in enumerate(sorted(node_freqs.items(), key=lambda kv: kv[1], reverse=True)):
        node_vocab[n] = i + 1

    return [subtoken_vocab, node_vocab]


def preprocess(arguments):
    pid, solutions, max_contexts_to_sample, [subtoken_vocab, node_vocab], args = arguments
    data = {}

    for solution in solutions:
        c2s = load_c2s(os.path.join(args.c2s_dir, pid, solution + '.c2s'))

        if len(c2s) > max_contexts_to_sample:
            np.random.shuffle(c2s)
            c2s = c2s[:max_contexts_to_sample]

        padded_paths = []
        path_lengths = []
        subtoken_ranges = []
        subtokens = []
        subtoken_idx = 0

        for ll, path, rl in c2s:
            lend = subtoken_idx + len(ll)
            rend = lend + len(rl)
            subtoken_ranges.append((subtoken_idx, lend, rend))
            subtoken_idx = rend
            ll_ids = np.asarray([subtoken_vocab.get(st, 0) for st in ll])
            rl_ids = np.asarray([subtoken_vocab.get(st, 0) for st in rl])
            subtokens.append(ll_ids)
            subtokens.append(rl_ids)
            padded_paths.append(np.asarray([node_vocab.get(n, 0) for n in path] + [0] * (args.max_path_length + 1 - len(path))))
            path_lengths.append(len(path))

        if len(c2s) > 0:
            subtokens = np.concatenate(subtokens)
            subtoken_ranges = np.asarray(subtoken_ranges)
            padded_paths = np.vstack(padded_paths)
            path_lengths = np.asarray(path_lengths)
        else:
            subtokens = np.asarray([0, 0])
            subtoken_ranges = np.asarray([[0, 1, 2]])
            padded_paths = np.asarray([[0] * (args.max_path_length + 1)])
            path_lengths = np.asarray([1])

        data[solution] = subtokens, subtoken_ranges, padded_paths, path_lengths

    return pid, data


def preprocess_dataset(split, vocab, args):
    def gen_args(ds_split, max_contexts_to_sample):
        for pid, sols in ds_split.items():
            yield pid, sols, max_contexts_to_sample, vocab, args

    dataset = {}

    with multiprocessing.Pool(args.num_workers) as pool:
        for pid, data in pool.imap_unordered(preprocess, gen_args(split[0], args.max_data_contexts)):
            dataset[pid] = data
        for pid, data in pool.imap_unordered(preprocess, gen_args(split[1], args.max_contexts)):
            pid_data = dataset.get(pid, None)
            if pid_data is None:
                pid_data = {}
                dataset[pid] = pid_data
            pid_data.update(data)
        for pid, data in pool.imap_unordered(preprocess, gen_args(split[2], args.max_contexts)):
            pid_data = dataset.get(pid, None)
            if pid_data is None:
                pid_data = {}
                dataset[pid] = pid_data
            pid_data.update(data)

    return dataset


def main():
    args = parse_args()

    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.split, 'rb') as f:
        split = pickle.load(f)

    vocab = build_vocab(split, args)

    with open(os.path.join(args.output_dir, 'vocab.pkl'), 'wb') as f:
        pickle.dump(vocab, f)

    dataset = preprocess_dataset(split, vocab, args)

    with open(os.path.join(args.output_dir, 'dataset.pkl'), 'wb') as f:
        pickle.dump(dataset, f)


if __name__ == "__main__":
    main()
