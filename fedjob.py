from nvflare import FedJob, FedAvg, ScriptExecutor
from torch.testing._internal.data.network1 import Net

if __name__ == "__main__":
    n_clients = 2
    num_rounds = 2
    train_script = "/Users/49751124/PycharmProjects/nvflare/NVFlare/examples/getting_started/pt/src/cifar10_fl.py"

    job = FedJob(name="cifar10_fedavg")

    # Define the controller workflow and send to server
    controller = FedAvg(
        num_clients=n_clients,
        num_rounds=num_rounds,
    )
    job.to_server(controller)

    # Define the initial global model and send to server
    job.to_server(Net())

    # Send executor to all clients
    executor = ScriptExecutor(
        task_script_path=train_script, task_script_args=""  # f"--batch_size 32 --data_path /tmp/data/site-{i}"
    )
    job.to_clients(executor)

    job_dir = 'jobs/job_config'
    job.export_job(job_dir)
    job.simulator_run(job_dir, n_clients=n_clients)
