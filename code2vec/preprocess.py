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
    parser.add_argument('--c2v-dir', '-i', type=str, required=True)
    parser.add_argument('--output-dir', '-o', type=str, required=True)
    parser.add_argument('--split', '-s', type=str, required=True)

    parser.add_argument('--min-leaf-count', '-lc', type=int, default=100)
    parser.add_argument('--min-path-count', '-pc', type=int, default=50)
    parser.add_argument('--max-contexts', '-cc', type=int, default=200)

    parser.add_argument('--num-workers', '-p', type=int,
                        default=os.cpu_count())
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def load_c2v(filename):
    c2v = []
    with open(filename) as f:
        for line in f:
            line = line.strip().split(' ')
            for s in line[1:]:
                ll, path, rl = s.split(',')
                c2v.append((ll, path, rl))
    return c2v


def get_freqs(arguments):
    pid, training_solutions, args = arguments
    leaf_freqs = defaultdict(int)
    path_freqs = defaultdict(int)
    for train_sol in training_solutions:
        c2v = load_c2v(os.path.join(args.c2v_dir, pid, train_sol + '.c2v'))
        for ll, path, rl in c2v:
            leaf_freqs[ll] += 1
            leaf_freqs[rl] += 1
            path_freqs[path] += 1
    return leaf_freqs, path_freqs


def build_vocab(split, args):
    training_set = split[0]

    def gen_args():
        for pid, training_solutions in training_set.items():
            yield pid, training_solutions, args

    leaf_freqs = defaultdict(int)
    path_freqs = defaultdict(int)
    with multiprocessing.Pool(args.num_workers) as pool:
        for leaf_freqs_local, path_freqs_local in pool.imap_unordered(
            get_freqs, gen_args(), chunksize=10
        ):
            for k, v in leaf_freqs_local.items():
                leaf_freqs[k] += v
            for k, v in path_freqs_local.items():
                path_freqs[k] += v

    all_freqs = (leaf_freqs, path_freqs)
    min_freqs = (args.min_leaf_count, args.min_path_count)
    vocabs = []
    for freqs, min_freq in zip(all_freqs, min_freqs):
        vocab_count_list = sorted(
            freqs.items(), key=lambda kv: kv[1], reverse=True)
        total = sum(map(lambda wc: wc[1], vocab_count_list))
        in_vocab = 0
        vocab = {'': 0}  # UNK
        for i, (word, count) in enumerate(vocab_count_list):
            if count < min_freq:
                break
            vocab[word] = i + 1
            in_vocab += count
        vocabs.append(vocab)
        print(f'Vocab size: {len(vocab)}/{len(vocab_count_list)}')
        print(f'Vocab coverage: {in_vocab}/{total} = {in_vocab/total:.2%}')

    return vocabs


def preprocess(arguments):
    pid, solutions, [leaf_vocab, path_vocab], args = arguments

    data = {}

    for solution in solutions:
        c2v = load_c2v(os.path.join(args.c2v_dir, pid, solution + '.c2v'))

        full_contexts = []
        partial_contexts = []
        for ll, path, rl in c2v:
            ll_idx = leaf_vocab.get(ll, 0)
            rl_idx = leaf_vocab.get(rl, 0)
            path_idx = path_vocab.get(path, 0)
            if ll_idx > 0 and rl_idx > 0 and path_idx > 0:
                full_contexts.append(np.asarray([ll_idx, path_idx, rl_idx]))
            elif ll_idx > 0 or rl_idx > 0 or path_idx > 0:
                partial_contexts.append(np.asarray([ll_idx, path_idx, rl_idx]))

        np.random.shuffle(full_contexts)
        np.random.shuffle(partial_contexts)

        if len(full_contexts) >= args.max_contexts:
            contexts = full_contexts[:args.max_contexts]
        elif len(full_contexts) + len(partial_contexts) >= args.max_contexts:
            contexts = full_contexts + \
                partial_contexts[:args.max_contexts-len(full_contexts)]
        else:
            contexts = full_contexts + partial_contexts

        if len(contexts) == 0:
            contexts = [np.asarray([0, 0, 0])]

        data[solution] = np.vstack(contexts)

    return pid, data


def merge_splits(split):
    merged_split = {}
    for ds in split:
        for pid, solutions in ds.items():
            psols = merged_split.get(pid, None)
            if psols is None:
                psols = []
                merged_split[pid] = psols
            psols += solutions
    return merged_split


def preprocess_dataset(split, vocab, args):
    def gen_args():
        for pid, sols in merge_splits(split).items():
            yield pid, sols, vocab, args

    dataset = {}
    with multiprocessing.Pool(args.num_workers) as pool:
        for pid, data in pool.imap_unordered(preprocess, gen_args()):
            dataset[pid] = data
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
