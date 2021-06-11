import os
import argparse
import numpy as np
import pickle
import random


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', '-i', type=str, required=True)
    parser.add_argument('--mode', '-m', type=str,
                        choices=('poj', 'gcj', 'i'), required=True)
    parser.add_argument('--train', '-t', type=str, nargs='+', required=False, default=None)
    parser.add_argument('--output', '-o', type=str, required=True)
    parser.add_argument('--seed', '-s', type=int, default=0)
    return parser.parse_args()


def collect_data(data_dir):
    data = {}
    for problem in os.listdir(data_dir):
        problem_dir = os.path.join(data_dir, problem)
        if os.path.isdir(problem_dir):
            problem_data = []
            data[problem] = problem_data
            for solution in os.listdir(problem_dir):
                path = os.path.join(problem_dir, solution)
                if os.path.isfile(path):
                    assert os.path.getsize(path) > 0
                    p = solution.rfind('.')
                    assert p > 0
                    problem_data.append(solution[:p])
    return data


def main():
    args = parse_args()

    random.seed(args.seed)

    data = collect_data(args.data_dir)

    datasets = [{} for _ in range(3)]
    if args.mode == 'poj':
        p = [0, 64, 80, 104]
        for i in range(3):
            for pid in range(p[i], p[i+1]):
                datasets[i][str(pid+1)] = data[str(pid+1)]
    elif args.mode == 'gcj':
        problem_list = sorted(data.items())
        random.shuffle(problem_list)
        problem_num = len(data)
        train_num = int(problem_num * 0.8)
        val_num = int(problem_num * 0.1)
        p = [0, train_num, train_num + val_num, problem_num]
        for i in range(3):
            for j in range(p[i], p[i+1]):
                pid, v = problem_list[j]
                datasets[i][pid] = v
    elif args.mode == 'i':
        assert args.train is not None
        p = [0, 64, 80, 104]
        for i in range(1, 3):
            for pid in range(p[i], p[i+1]):
                datasets[i][str(pid+1)] = data[str(pid+1)]
        for raw_pid in args.train:
            datasets[0][raw_pid] = data[raw_pid]
    else:
        raise Exception

    print(f'Number of problems: {len(datasets[0])}, {len(datasets[1])}, {len(datasets[2])}')
    print(f'[train, val, test]:',
          list(map(lambda x: sum(map(len, x.values())), datasets)))

    with open(args.output, 'wb') as f:
        pickle.dump(datasets, f)


if __name__ == "__main__":
    main()
