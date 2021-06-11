import argparse
import os
import pickle
from collections import defaultdict
import numpy as np
import multiprocessing

from cass import load_file


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cass-dir', '-i', type=str, required=True)
    parser.add_argument('--output-dir', '-o', type=str, required=True)
    parser.add_argument('--split', '-s', type=str, required=True)
    parser.add_argument('--num-workers', '-p', type=int,
                        default=os.cpu_count())
    return parser.parse_args()


def load_feature(arguments):
    pid, problem_dir, solution = arguments
    casss = load_file(os.path.join(problem_dir, solution))
    assert len(casss) > 0
    features = defaultdict(int)
    for cass in casss:
        for feat in cass.featurize():
            features[feat] += 1
    return pid, solution, features


def load_features(args):
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

    features = defaultdict(dict)
    with multiprocessing.Pool(args.num_workers) as pool:
        for pid, solution, feature in pool.imap_unordered(load_feature, paths, chunksize=10):
            if feature is not None:
                features[pid][solution[:-4]] = feature

    return dict(features)


def get_freqs(arguments):
    problem_features, training_solutions = arguments
    freqs = defaultdict(int)
    for train_sol in training_solutions:
        feature_counts = problem_features[train_sol]
        for k, v in feature_counts.items():
            freqs[k] += v
    return freqs


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


def build_vocab(features, split, args):
    training_set = merge_splits(split)

    def gen_args():
        for pid, training_solutions in training_set.items():
            yield features[pid], training_solutions

    freqs = defaultdict(int)
    with multiprocessing.Pool(args.num_workers) as pool:
        for freqs_local in pool.imap_unordered(get_freqs, gen_args(), chunksize=10):
            for k, v in freqs_local.items():
                freqs[k] += v

    vocab_count_list = sorted(
        freqs.items(), key=lambda kv: kv[1], reverse=True)
    vocab = {}
    for i, (word, count) in enumerate(vocab_count_list):
        vocab[word] = i

    print(f'Vocab size: {len(vocab)}/{len(vocab_count_list)}')

    return vocab


def preprocess(arguments):
    pid, problem_features, vocab = arguments
    data = {}
    for solution, feature_counts in problem_features.items():
        feature_ids = []
        counts = []
        for ft, cnt in feature_counts.items():
            fid = vocab.get(ft, -1)
            if fid >= 0:
                feature_ids.append(fid)
                counts.append(cnt)
        feature_ids = np.asarray(feature_ids)
        counts = np.asarray(counts)
        data[solution] = (feature_ids, counts)
    return pid, data


def preprocess_dataset(features, vocab, args):
    dataset = {}
    def gen_args():
        for i, fs in features.items():
            yield i, fs, vocab
    with multiprocessing.Pool(args.num_workers) as pool:
        for pid, data in pool.imap_unordered(preprocess, gen_args()):
            dataset[pid] = data
    return dataset


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.split, 'rb') as f:
        split = pickle.load(f)

    features = load_features(args)

    vocab = build_vocab(features, split, args)

    with open(os.path.join(args.output_dir, 'vocab.pkl'), 'wb') as f:
        pickle.dump(vocab, f)

    dataset = preprocess_dataset(features, vocab, args)

    with open(os.path.join(args.output_dir, 'dataset.pkl'), 'wb') as f:
        pickle.dump(dataset, f)


if __name__ == "__main__":
    main()
