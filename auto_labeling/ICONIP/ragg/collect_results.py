"""

    python3 collect_results.py -i 'log' -j 246750 -n 25
"""

import argparse


# Define the function to parse the parameters
def parse_arguments():
    parser = argparse.ArgumentParser(description="FedGNN")

    # Add arguments to be parsed
    parser.add_argument('-i', '--in_dir', type=str, required=False, default='log',
                        help="input directory")
    parser.add_argument('-j', '--job_id', type=int, required=False, default=246750,
                        help="Job ID")
    parser.add_argument('-n', '--num_tasks', type=int, required=False, default=25,
                        help="num_tasks")
    # Parse the arguments
    args = parser.parse_args()

    # Return the parsed arguments
    return args


# Parse command-line arguments
args = parse_arguments()

in_dir = args.in_dir
job_id = args.job_id
num_tasks = args.num_tasks

print(args)


def collect():
    out_file = f'{job_id}.txt'
    with open(out_file, 'w+') as out_f:
        for i in range(num_tasks):
            in_file = f'{in_dir}/output_{job_id}_{i}.out'
            print(in_file)
            out_f.write('\n\n')
            out_f.write(in_file + '\n')

            try:
                with open(in_file, 'r') as in_f:
                    data = in_f.readlines()

                first_lines = min(5, len(data) - 1)
                # print(first_lines, len(data), flush=True)
                for i in range(first_lines):
                    try:
                        # print(i, data[i:(i+1)], flush=True)
                        line = data[i:(i + 1)][0]
                    except Exception as e:
                        # print(i, e, flush=True)
                        line = '\n'
                    # print(line)
                    out_f.write(line)

                # print(data[-30:])
                for i in range(min(len(data), 31), 0, -1):
                    # print(i, data[-1 * i: (-1*(i-1))], flush=True)
                    try:
                        line = data[-1 * i: (-1 * (i - 1))][0]
                    except Exception as e:
                        # print(i, e, flush=True)
                        line = '\n'
                    # print(line)
                    out_f.write(line)

            except Exception as e:
                print(e)


if __name__ == '__main__':
    collect()
