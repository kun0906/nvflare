"""

"""
import os.path
import shutil

print(os.getcwd())
# os.chdir('/Users/kun/Projects/nvflare/cifar10-hello-pt-10clients-2classes')
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
    root_dir = 'jobs/10clients-2classes_byzantine'
    data_type = "normal"     # "attack_black_all"    #'normal'
    src = os.path.join(root_dir, 'app_site-template')

    # normal clients  and byzantine clients
    n_byzantine_clients = 2
    for i in range(1, n_clients + 1):
        dst = os.path.join(root_dir, 'app_site-' + str(i))
        print(f'\nsite {i}: copy scr:{src} -> dst:{dst}')
        shutil.copytree(src, dst, dirs_exist_ok=True)

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

        if i <= n_byzantine_clients:
            train_file = os.path.join(dst, 'custom/learner_with_tb.py')
            ith_line = 239 - 1  # byzantine clients send very large number to server
            #   new_weights = {k: v.cpu().numpy() for k, v in new_weights.items()}
            new_line = "        new_weights = {k: v.cpu().numpy().fill(1000) for k, v in new_weights.items()}\n"
            replace_line(train_file, ith_line, new_line)


if __name__ == '__main__':
    main()
