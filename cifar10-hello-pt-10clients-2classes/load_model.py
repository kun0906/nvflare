import os
import torch


def load_model():
    # model_dir = '/Users/49751124/cifar10-hello-pt-10clients-2classes/transfer/55f6726c-ea74-4735-b710-894d6c68be8e/workspace/app_server/models'
    model_dir = '/Users/kun/Projects/nvflare/poc/example_project/prod_00/admin@nvidia.com/transfer/8adf33e3-fbda-4676-b719-c03278da06d4/workspace/app_server/models'
    model_dir = '/Users/kun/Projects/nvflare/poc/example_project/prod_00/admin@nvidia.com/transfer/7d162b77-d9fe-46fd-a005-f6717b1b6cee/workspace/app_server/models'

    if not os.path.exists(model_dir):
        print(f'model_dir does not exist)')
    # Load the model with weights_only=True
    # model = torch.load(f'{model_dir}/global_weights_0.pkl', weights_only=False)
    # print(os.path.exists(f'{model_dir}'))
    model = torch.load(f'{model_dir}/site-1_weights_8.pkl', weights_only=False)
    print(model)

    for site, data in model.items():
        print(site, data)


if __name__ == '__main__':
    load_model()
