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
