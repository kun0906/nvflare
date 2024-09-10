"""

"""
import os.path
import shutil

print(os.getcwd())
os.chdir('/Users/kun/Projects/nvflare/cifar10-hello-pt-10clients-2classes')
# os.chdir('/Users/49751124/PycharmProjects/nvflare/cifar10-hello-pt-10clients-2classes')
print(os.getcwd())


def replace_line(dst, i, new_line=''):
    with open(dst, 'r') as file:
        lines = file.readlines()
    lines[i] = new_line

    with open(dst, 'w') as file:
        file.writelines(lines)


def main():
    n_clients = 10
    # root_dir = 'jobs/hello-pt-10clients-2classes'
    root_dir = 'jobs/10clients-2classes'
    src = os.path.join(root_dir, 'app_site-template')
    for i in range(1, n_clients+1):
        print(f'\nsite {i}:')
        dst = os.path.join(root_dir, 'app_site-' + str(i))
        shutil.copytree(src, dst, dirs_exist_ok=True)

        remote_dir = '/users/kunyang'
        # remote_dir = '/Users/kun/Projects/nvflare'
        # replace the content
        train_file = os.path.join(dst, 'custom/learner_with_tb.py')
        ith_line = 50-1
        new_line = f"        train_path='{remote_dir}/cifar10-hello-pt-10clients-2classes/data/client_{i}_airplane_train.pkl',\n"
        replace_line(train_file, ith_line, new_line)

        valid_file = os.path.join(dst, 'custom/learner_with_tb.py')
        ith_line = 51-1
        new_line = f"        test_path='{remote_dir}/cifar10-hello-pt-10clients-2classes/data/client_{i}_airplane_test.pkl',\n"
        replace_line(valid_file, ith_line, new_line)


if __name__ == '__main__':
    main()
