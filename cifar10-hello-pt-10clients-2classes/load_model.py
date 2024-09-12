import torch


def load_model():
    model_dir = '/Users/49751124/cifar10-hello-pt-10clients-2classes/transfer/55f6726c-ea74-4735-b710-894d6c68be8e/workspace/app_server/models'

    # Load the model with weights_only=True
    model = torch.load(f'{model_dir}/global_weights_0.pkl', weights_only=False)

    print(model)


if __name__ == '__main__':
    load_model()
