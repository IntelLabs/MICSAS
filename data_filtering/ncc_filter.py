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
import shutil


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ncc-dir', '-n', type=str, required=True)
    parser.add_argument('--input-dir', '-i', type=str, required=True)
    parser.add_argument('--output-dir', '-o', type=str, required=True)
    parser.add_argument('--num-workers', '-p', type=int,
                        default=os.cpu_count())
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    np = 0
    ns = 0

    for d in os.listdir(args.ncc_dir):
        if d.startswith('seq_'):
            pid = d[4:]
            outd = os.path.join(args.output_dir, pid)
            os.makedirs(outd, exist_ok=True)
            np += 1
            for fn in os.listdir(os.path.join(args.ncc_dir, d)):
                if fn.endswith('.csv'):
                    sid = fn[:-8]
                    shutil.move(os.path.join(args.input_dir, pid, f'{sid}.cpp'), os.path.join(outd, f'{sid}.cpp'))
                    ns += 1

    print(np, ns)


if __name__ == "__main__":
    main()
