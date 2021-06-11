import argparse
import os
import pickle
import numpy as np
from scipy.sparse import csr_matrix, vstack
from sklearn.metrics import pairwise
import torch

from train import compute_metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', '-f', type=str, required=True)
    parser.add_argument('--split', type=str, required=True)
    parser.add_argument('--sim', type=str, choices=('dot', 'cos'), required=True)
    return parser.parse_args()


def get_code_vecs(dataset, vocab, split):
    data = [[] for _ in range(len(split))]
    for i, (pid, solutions) in enumerate(split.items()):
        problem_data = dataset[pid]
        problem_split_data = data[i]
        for sol in solutions:
            problem_split_data.append(problem_data[sol])

    code_vecs = []
    pids = []
    for pid in range(len(data)):
        solutions = data[pid]
        for sid in range(len(solutions)):
            features, counts = solutions[sid]

            vec = csr_matrix(
                (
                    np.ones(len(features), dtype=np.int32),
                    (np.zeros(len(features), dtype=np.int32), features)
                ),
                shape=(1, len(vocab))
            )
            code_vecs.append(vec)

            pids.append(pid)

    code_vecs = vstack(code_vecs)
    pids = np.array(pids)

    return code_vecs, pids


def main():
    args = parse_args()

    with open(args.split, 'rb') as f:
        split = pickle.load(f)

    with open(os.path.join(args.dataset_dir, 'vocab.pkl'), 'rb') as f:
        vocab = pickle.load(f)

    with open(os.path.join(args.dataset_dir, 'dataset.pkl'), 'rb') as f:
        dataset = pickle.load(f)
    
    code_vecs, pids = get_code_vecs(dataset, vocab, split[2])

    if args.sim == 'dot':
        sim = (code_vecs @ code_vecs.T).toarray().astype(np.float32)
    elif args.sim == 'cos':
        sim = pairwise.cosine_similarity(code_vecs)
    else:
        raise Exception

    compute_metrics(torch.from_numpy(sim), torch.from_numpy(pids))


if __name__ == "__main__":
    main()
