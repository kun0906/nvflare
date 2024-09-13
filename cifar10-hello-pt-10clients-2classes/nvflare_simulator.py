"""Debug your code with nvflare simulator


"""

import argparse
import sys
from sys import platform

from nvflare.private.fed.app.simulator.simulator_runner import SimulatorRunner

JOB_NAME = "10clients-2classes"


def define_simulator_parser(simulator_parser):
    simulator_parser.add_argument("-jf", "--job_folder", default=f"jobs/{JOB_NAME}", type=str, help="WORKSPACE folder")
    simulator_parser.add_argument("-w", "--workspace", default=f"simulator/{JOB_NAME}", type=str,
                                  help="WORKSPACE folder")
    simulator_parser.add_argument("-n", "--n_clients", default=3, type=int, help="number of clients")
    simulator_parser.add_argument("-c", "--clients", default="", type=str, help="client names list")
    simulator_parser.add_argument("-t", "--threads", default=2, type=int, help="number of parallel running clients")
    simulator_parser.add_argument("-gpu", "--gpu", default=0, type=str, help="list of GPU Device Ids, comma separated")
    simulator_parser.add_argument("-m", "--max_clients", type=int, default=100, help="max number of clients")


def run_simulator(simulator_args):
    print(simulator_args)
    simulator = SimulatorRunner(
        job_folder=simulator_args.job_folder,
        workspace=simulator_args.workspace,
        clients=simulator_args.clients,
        n_clients=simulator_args.n_clients,
        threads=simulator_args.threads,
        gpu=simulator_args.gpu,
        max_clients=simulator_args.max_clients,
    )
    run_status = simulator.run()

    return run_status


if __name__ == "__main__":
    """
    This is the main program when running the NVFlare Simulator. Use the Flare simulator API,
    create the SimulatorRunner object, do a setup(), then calls the run().
    """

    if sys.version_info < (3, 7):
        raise RuntimeError("Please use Python 3.7 or above.")

    parser = argparse.ArgumentParser()
    define_simulator_parser(parser)
    args = parser.parse_args()
    status = run_simulator(args)
    sys.exit(status)
