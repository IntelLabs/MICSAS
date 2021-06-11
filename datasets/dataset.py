from abc import ABC, abstractmethod
import numpy as np
import torch


class Dataset(ABC):
    @abstractmethod
    def get_dataset(self):
        pass

    @abstractmethod
    def collate(self, batch):
        pass

    def _get_data_split(self, split):
        dataset = self.get_dataset()
        data = [[] for _ in range(len(split))]
        pids = sorted(split.keys())
        for i, pid in enumerate(pids):
            solutions = split[pid]
            problem_data = dataset[pid]
            problem_split_data = data[i]
            for sol in solutions:
                if sol not in problem_data:
                    for k, v in problem_data.items():
                        if k.split('-')[0] == sol:
                            problem_split_data.append((v, i))
                else:
                    problem_split_data.append((problem_data[sol], i))
        return data

    def get_pk_sample_generator_function(self, split, p, k):
        data = self._get_data_split(split)

        def gen():
            while True:
                pids = np.random.choice(len(data), p, replace=False)
                batch = []
                labels = []
                for pid in pids:
                    solutions = data[pid]
                    solution_num = len(solutions)
                    p_sample_num = solution_num if solution_num < k else k
                    sids = np.random.choice(solution_num, p_sample_num, replace=False)
                    for sid in sids:
                        sol_data, _ = solutions[sid]
                        batch.append(sol_data)
                        labels.append(pid)
                yield self.collate(batch), torch.tensor(labels)

        return gen

    def get_data_generator_function(self, split, batch_size, shuffle=False):
        data = []
        for solutions in self._get_data_split(split):
            data += solutions

        num_batches = len(data) // batch_size
        if len(data) % batch_size > 0:
            num_batches += 1

        def gen():
            iter_data = data.copy()
            if shuffle:
                np.random.shuffle(iter_data)
            labels = []
            batch = []
            for sol_data, label_id in iter_data:
                batch.append(sol_data)
                labels.append(label_id)
                if len(batch) == batch_size:
                    yield self.collate(batch), torch.tensor(labels)
                    labels = []
                    batch = []
            if len(batch) > 0:
                yield self.collate(batch), torch.tensor(labels)

        return gen, num_batches
