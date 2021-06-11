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
    parser.add_argument('--min-freq', '-t', type=int, default=5)
    parser.add_argument('--load-vocab', '-l', type=str, default=None)
    parser.add_argument('--num-workers', '-p', type=int,
                        default=os.cpu_count())
    return parser.parse_args()


def build_graph(node, nodes, edges):
    node_id = len(nodes)
    nodes.append(node.n)
    last_id = node_id
    for c in node.children:
        edges.append((node_id, last_id + 1))
        last_id = build_graph(c, nodes, edges)
    return last_id


def load_graph(arguments):
    pid, problem_dir, solution = arguments
    casss = load_file(os.path.join(problem_dir, solution))
    assert len(casss) > 0
    nodes = []
    edges = []
    for cass in casss:
        build_graph(cass.root, nodes, edges)
    return pid, solution, (nodes, edges)


def load_graphs(args):
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

    node_nums = []
    edge_nums = []

    graphs = defaultdict(dict)
    with multiprocessing.Pool(args.num_workers) as pool:
        for pid, solution, graph in pool.imap_unordered(load_graph, paths, chunksize=10):
            if graph is not None:
                graphs[pid][solution[:-4]] = graph
                node_nums.append(len(graph[0]))
                edge_nums.append(len(graph[1]))

    print(f'Mean #nodes: {np.mean(node_nums)}, mean #edes: {np.mean(edge_nums)}')

    return dict(graphs)


def get_freqs(arguments):
    problem_graphs, training_solutions = arguments
    freqs = defaultdict(int)
    for train_sol in training_solutions:
        nodes, _ = problem_graphs[train_sol]
        for t in nodes:
            freqs[t] += 1
    return freqs


def build_vocab(graphs, split, args):
    training_set = split[0]

    def gen_args():
        for pid, training_solutions in training_set.items():
            yield graphs[pid], training_solutions

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
        if count < args.min_freq:
            break
        vocab[word] = i + 1  # Reserve 0 for UNK
        in_vocab += count

    vocab[''] = 0

    print(f'Vocab size: {len(vocab)}/{len(vocab_count_list)}')
    print(f'Vocab coverage: {in_vocab}/{total} = {in_vocab/total:.2%}')

    return vocab


def preprocess(arguments):
    pid, problem_graphs, vocab = arguments
    data = {}
    for solution, (nodes, edges) in problem_graphs.items():
        nodes = np.asarray([vocab.get(t, 0) for t in nodes])
        edges = np.asarray(edges).T
        data[solution] = (nodes, edges)
    return pid, data


def preprocess_dataset(graphs, vocab, args):
    dataset = {}
    def gen_args():
        for i, g in graphs.items():
            yield i, g, vocab
    with multiprocessing.Pool(args.num_workers) as pool:
        for pid, data in pool.imap_unordered(preprocess, gen_args()):
            dataset[pid] = data
    return dataset


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.split, 'rb') as f:
        split = pickle.load(f)

    graphs = load_graphs(args)

    if args.load_vocab is None:
        vocab = build_vocab(graphs, split, args)
    else:
        with open(os.path.join(args.load_vocab), 'rb') as f:
            vocab = pickle.load(f)

    with open(os.path.join(args.output_dir, 'vocab.pkl'), 'wb') as f:
        pickle.dump(vocab, f)

    dataset = preprocess_dataset(graphs, vocab, args)

    with open(os.path.join(args.output_dir, 'dataset.pkl'), 'wb') as f:
        pickle.dump(dataset, f)


if __name__ == "__main__":
    main()
