"""

"""
import os.path
import shutil

print(os.getcwd())
# os.chdir('/Users/kun/Projects/nvflare/cifar10-hello-pt-10clients-2classes')
<<<<<<< HEAD
# os.chdir('/Users/49751124/PycharmProjects/nvflare/cifar10-hello-pt-10clients-2classes')
=======
# os.chdir('/Users/kun/Projects/nvflare/cifar10-hello-pt-10clients-2classes')
os.chdir('/Users/49751124/PycharmProjects/nvflare/cifar10-hello-pt-10clients-2classes')
>>>>>>> 45127c9 (v0.0.7-1:sync with different devices)
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
    data_type = "attack_black_all" #'normal'
    src = os.path.join(root_dir, 'app_site-template')
    for i in range(1, n_clients + 1):
        print(f'\nsite {i}:')
        dst = os.path.join(root_dir, 'app_site-' + str(i))
        shutil.copytree(src, dst, dirs_exist_ok=True)

<<<<<<< HEAD
        # remote_dir = '/users/kunyang'
        # remote_dir = '/Users/kun/Projects/nvflare'
<<<<<<< HEAD
=======
        # remote_dir = '/Users/49751124/PycharmProjects/nvflare'
>>>>>>> 45127c9 (v0.0.7-1:sync with different devices)
=======
>>>>>>> f811d76 (v0.1.3: Change input data location to "~/data/data_type")
        # replace the content
        train_file = os.path.join(dst, 'custom/learner_with_tb.py')
        ith_line = 54 - 1
        new_line = f"            train_path='~/data/{data_type}/client_{i}_airplane_train.pkl',\n"
        replace_line(train_file, ith_line, new_line)

        valid_file = os.path.join(dst, 'custom/learner_with_tb.py')
        ith_line = 55 - 1
        new_line = f"            test_path='~/data/{data_type}/client_{i}_airplane_test.pkl',\n"
        replace_line(valid_file, ith_line, new_line)

        train_file = os.path.join(dst, 'custom/learner_with_tb.py')
        ith_line = 56 - 1
        site_id = i
        new_line = f"            site_id={site_id},\n"
        replace_line(train_file, ith_line, new_line)


if __name__ == '__main__':
    main()