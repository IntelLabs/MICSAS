import argparse
import multiprocessing
import os
import tqdm
import subprocess


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--extractor', '-e', type=str, required=False,
                        default='./cass-extractor/build/bin/cass-extractor')
    parser.add_argument('--input-dir', '-i', type=str, required=True,
                        help='The dataset directory')
    parser.add_argument('--output-dir', '-o', type=str, required=True,
                        help='The output directory')
    parser.add_argument('--num-workers', '-p', type=int, default=os.cpu_count(),
                        help='Number of workers to use')
    args = parser.parse_args()
    return args


def extract_from_solution(arguments):
    problem, solution, args = arguments

    os.makedirs(os.path.join(args.output_dir, problem), exist_ok=True)

    solution_path = os.path.join(
        args.input_dir,
        problem,
        solution
    )

    output_path = os.path.join(
        args.output_dir,
        problem,
        solution[:solution.rfind('.')] + '.cas'
    )

    try:
        output = subprocess.check_output([args.extractor, '-f', solution_path])
        with open(output_path, 'w') as outf:
            for line in output.decode('utf-8').splitlines():
                if len(line) >= 0:
                    outf.write(line)
                    outf.write('\n')
    except subprocess.CalledProcessError:
        return problem, solution

    return None


def main():
    args = parse_args()

    solution_list = []
    for problem in os.listdir(args.input_dir):
        problem_dir = os.path.join(args.input_dir, problem)
        for solution in os.listdir(problem_dir):
            solution_path = os.path.join(problem_dir, solution)
            solution_list.append((problem, solution))

    def gen_args():
        for problem, solution in solution_list:
            yield problem, solution, args

    err_solutions = []
    with multiprocessing.Pool(args.num_workers) as pool:
        for r in tqdm.tqdm(
            pool.imap_unordered(
                extract_from_solution,
                gen_args()
            ),
            total=len(solution_list)
        ):
            if r is not None:
                err_solutions.append(r)

    print(f'Error solutions: {len(err_solutions)}')


if __name__ == "__main__":
    main()
