from tensorboard.backend.event_processing import event_accumulator


def read_tensorboard_events(log_dir):
    print(log_dir)
    # Create an EventAccumulator instance
    event_acc = event_accumulator.EventAccumulator(log_dir)
    event_acc.Reload()

    # Get scalar tags
    scalar_tags = event_acc.Tags()['scalars']
    print('Scalar Tags:', scalar_tags)
    # Read scalar data
    scalars = {}
    for tag in scalar_tags:
        events = event_acc.Scalars(tag)
        scalars[tag] = [(e.step, e.value) for e in events]

    return scalars


def main():
    # Define the path to your TensorBoard log directory
    root_dir = '/Users/49751124/cifar10-hello-pt-10clients-2classes/transfer'
    job_id = '228b24b6-400e-485e-a733-4e1444278615'
    log_dir = f'{root_dir}/{job_id}/workspace/tb_events/site-1'
    scalars = read_tensorboard_events(log_dir)

    # Print out all the results
    for tag, values in scalars.items():
        print(f'Tag: {tag}')
        for step, value in values:
            print(f'Step: {step}, Value: {value}')


if __name__ == '__main__':
    main()
