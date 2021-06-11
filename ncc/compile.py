import argparse
import os
import multiprocessing
import subprocess
import tqdm
import pickle
import tempfile
from collections import defaultdict
from absl import flags
import wget
import zipfile
import shutil
import contextlib
import re

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import sys
sys.path.append(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'ncc'))
import task_utils


FLAGS = flags.FLAGS

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
    parser.add_argument('--input-dir', '-i', type=str, required=True)
    parser.add_argument('--output-dir', '-o', type=str, required=True)
    parser.add_argument('--split', '-s', type=str, required=True)
    parser.add_argument('--dataset', '-d', type=str, choices=('poj', 'gcj'),
                        required=True)
    parser.add_argument('--clangxx', '-c', type=str, default='clang++')
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--num-workers', '-p', type=int,
                        default=os.cpu_count())
    return parser.parse_args()


def compile_file(arguments):
    pid, sol_name, augment, args = arguments

    ir_dir = f'{args.output_dir}/ir_{pid}'
    os.makedirs(ir_dir, exist_ok=True)

    extension = 'txt' if args.dataset == 'poj' else 'cpp'
    input_file = f'{args.input_dir}/{pid}/{sol_name}.{extension}'

    with open(input_file, 'rb') as f:
        src = f.read()

    if args.dataset == 'poj':
        src = header + src.replace(b'void main', b'int main')

    failed = set()

    opts = []
    if augment:
        for opt in ['-O0', '-O1', '-O2', '-O3']:
            for fm in ['', '-ffast-math']:
                opts.append((opt, fm))
    else:
        opts = [('', '')]

    for opt, fm in opts:
        ir_tmp_dir = f'{ir_dir}/{sol_name}{opt}{fm}'
        os.makedirs(ir_tmp_dir, exist_ok=True)

        ir_file = f'{ir_tmp_dir}/{sol_name}{opt}{fm}.ll'

        cmd = f'{args.clangxx} -x c++ {flags} {opt} {fm} - -o /dev/stdout'
        try:
            p = subprocess.run(cmd.split(' '), check=True, input=src,
                               stdout=subprocess.PIPE)
        except subprocess.CalledProcessError:
            failed.add((pid, sol_name))
            shutil.rmtree(ir_tmp_dir)
            continue

        with open(ir_file, 'wb') as f:
            f.write(p.stdout)

        seq_tmp_dir = re.sub('ir', 'seq', ir_tmp_dir)
        seq_dir = os.path.split(seq_tmp_dir)[0]
        with contextlib.redirect_stdout(None):
            try:
                task_utils.llvm_ir_to_trainable(ir_tmp_dir)
            except Exception as e:
                print(e)
                failed.add((pid, sol_name))
                if os.path.exists(seq_tmp_dir):
                    shutil.rmtree(seq_tmp_dir)
            else:
                for fn in os.listdir(seq_tmp_dir):
                    if fn.endswith('.csv'):
                        shutil.move(os.path.join(seq_tmp_dir, fn), seq_dir)
                shutil.rmtree(seq_tmp_dir)

        shutil.rmtree(ir_tmp_dir)

    return failed


def compile_dataset(dataset, augment, args):
    tasks = []
    for pid, solutions in dataset.items():
        for sol in solutions:
            tasks.append((pid, sol, augment, args))

    failed_all = set()
    with multiprocessing.Pool(args.num_workers) as pool:
        for failed in tqdm.tqdm(pool.imap_unordered(compile_file, tasks), total=len(tasks)):
            failed_all.update(failed)

    print(f'Failed: {len(failed_all)}')


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    FLAGS(sys.argv[:1])
    FLAGS.vocabulary_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ncc', FLAGS.vocabulary_dir)
    if not os.path.exists(FLAGS.vocabulary_dir):
        with tempfile.TemporaryDirectory() as tempdir:
            vocab_zip = wget.download(
                'https://polybox.ethz.ch/index.php/s/AWKd60qR63yViH8/download', out=tempdir)
            zipfile.ZipFile(vocab_zip, 'r').extractall(tempdir)
            shutil.move(os.path.join(tempdir, 'vocabulary'),
                        FLAGS.vocabulary_dir)

    with open(args.split, 'rb') as f:
        split = pickle.load(f)

    for i, augment in zip(range(3), (args.augment, False, False)):
        compile_dataset(split[i], augment, args)


if __name__ == "__main__":
    main()
