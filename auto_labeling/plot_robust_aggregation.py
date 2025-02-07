import re

import matplotlib.pyplot as plt


def plot_robust_aggregation():
    # LABELING_RATE: 0.2, NUM_BENIGN_CLIENTS: 5, NUM_BYZANTINE_CLIENTS: 4, NUM_CLASSES: 10
    global_accs = {'refined_krum':
                       """
                        Epoch: 0, labeled_acc:0.12, val_acc:0.13, unlabeled_acc:0.13, shared_acc:0.14
                        Epoch: 1, labeled_acc:0.93, val_acc:0.93, unlabeled_acc:0.94, shared_acc:0.93
                        Epoch: 2, labeled_acc:0.98, val_acc:0.98, unlabeled_acc:0.97, shared_acc:0.92
                        Epoch: 3, labeled_acc:1.00, val_acc:0.98, unlabeled_acc:0.98, shared_acc:0.81
                        Epoch: 4, labeled_acc:0.96, val_acc:0.98, unlabeled_acc:0.96, shared_acc:0.95
                        Epoch: 5, labeled_acc:0.96, val_acc:0.97, unlabeled_acc:0.96, shared_acc:0.95
                        Epoch: 6, labeled_acc:0.96, val_acc:0.97, unlabeled_acc:0.96, shared_acc:0.95
                        Epoch: 7, labeled_acc:0.96, val_acc:0.97, unlabeled_acc:0.96, shared_acc:0.95
                        Epoch: 8, labeled_acc:0.96, val_acc:0.97, unlabeled_acc:0.96, shared_acc:0.95
                        Epoch: 9, labeled_acc:0.96, val_acc:0.98, unlabeled_acc:0.96, shared_acc:0.95
                       """,
                   'krum':
                       """
                        Epoch: 0, labeled_acc:0.12, val_acc:0.13, unlabeled_acc:0.13, shared_acc:0.14
                        Epoch: 1, labeled_acc:0.93, val_acc:0.93, unlabeled_acc:0.94, shared_acc:0.93
                        Epoch: 2, labeled_acc:0.98, val_acc:0.98, unlabeled_acc:0.97, shared_acc:0.92
                        Epoch: 3, labeled_acc:1.00, val_acc:0.98, unlabeled_acc:0.98, shared_acc:0.81
                        Epoch: 4, labeled_acc:0.96, val_acc:0.98, unlabeled_acc:0.96, shared_acc:0.95
                        Epoch: 5, labeled_acc:0.96, val_acc:0.97, unlabeled_acc:0.96, shared_acc:0.95
                        Epoch: 6, labeled_acc:0.96, val_acc:0.97, unlabeled_acc:0.96, shared_acc:0.95
                        Epoch: 7, labeled_acc:0.96, val_acc:0.97, unlabeled_acc:0.96, shared_acc:0.95
                        Epoch: 8, labeled_acc:0.96, val_acc:0.97, unlabeled_acc:0.96, shared_acc:0.95
                        Epoch: 9, labeled_acc:0.96, val_acc:0.98, unlabeled_acc:0.96, shared_acc:0.95
                       """,
                   'median':
                       """
                        Epoch: 0, labeled_acc:0.12, val_acc:0.13, unlabeled_acc:0.13, shared_acc:0.14
                        Epoch: 1, labeled_acc:0.19, val_acc:0.21, unlabeled_acc:0.20, shared_acc:0.10
                        Epoch: 2, labeled_acc:0.47, val_acc:0.52, unlabeled_acc:0.47, shared_acc:0.24
                        Epoch: 3, labeled_acc:0.19, val_acc:0.20, unlabeled_acc:0.20, shared_acc:0.22
                        Epoch: 4, labeled_acc:0.77, val_acc:0.80, unlabeled_acc:0.77, shared_acc:0.85
                        Epoch: 5, labeled_acc:0.69, val_acc:0.70, unlabeled_acc:0.70, shared_acc:0.40
                        Epoch: 6, labeled_acc:0.64, val_acc:0.71, unlabeled_acc:0.64, shared_acc:0.79
                        Epoch: 7, labeled_acc:0.93, val_acc:0.93, unlabeled_acc:0.93, shared_acc:0.57
                        Epoch: 8, labeled_acc:0.50, val_acc:0.56, unlabeled_acc:0.51, shared_acc:0.71
                        Epoch: 9, labeled_acc:0.98, val_acc:0.98, unlabeled_acc:0.97, shared_acc:0.53
                       """,
                   'mean':
                       """
                        Epoch: 0, labeled_acc:0.12, val_acc:0.13, unlabeled_acc:0.13, shared_acc:0.14
                        Epoch: 1, labeled_acc:0.06, val_acc:0.03, unlabeled_acc:0.06, shared_acc:0.08
                        Epoch: 2, labeled_acc:0.18, val_acc:0.17, unlabeled_acc:0.19, shared_acc:0.14
                        Epoch: 3, labeled_acc:0.06, val_acc:0.06, unlabeled_acc:0.07, shared_acc:0.13
                        Epoch: 4, labeled_acc:0.28, val_acc:0.26, unlabeled_acc:0.29, shared_acc:0.14
                        Epoch: 5, labeled_acc:0.14, val_acc:0.15, unlabeled_acc:0.12, shared_acc:0.10
                        Epoch: 6, labeled_acc:0.16, val_acc:0.16, unlabeled_acc:0.17, shared_acc:0.11
                        Epoch: 7, labeled_acc:0.08, val_acc:0.08, unlabeled_acc:0.07, shared_acc:0.11
                        Epoch: 8, labeled_acc:0.11, val_acc:0.09, unlabeled_acc:0.10, shared_acc:0.08
                        Epoch: 9, labeled_acc:0.13, val_acc:0.14, unlabeled_acc:0.14, shared_acc:0.12
                       """
                   }

    # LABELING_RATE: 0.8, NUM_BENIGN_CLIENTS: 4, NUM_BYZANTINE_CLIENTS: 3, NUM_CLASSES: 10
    # flip_labeling for attackers. data_poisoning
    global_accs2 = {'refined_krum':
                       """
                        Epoch: 0, labeled_acc:0.13, val_acc:0.12, unlabeled_acc:0.12, shared_acc:0.14
                        Epoch: 1, labeled_acc:0.92, val_acc:0.91, unlabeled_acc:0.93, shared_acc:0.90
                        Epoch: 2, labeled_acc:0.01, val_acc:0.01, unlabeled_acc:0.01, shared_acc:0.03
                        Epoch: 3, labeled_acc:0.97, val_acc:0.97, unlabeled_acc:0.98, shared_acc:0.97
                        Epoch: 4, labeled_acc:0.00, val_acc:0.00, unlabeled_acc:0.00, shared_acc:0.00
                        Epoch: 5, labeled_acc:0.98, val_acc:0.98, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 6, labeled_acc:0.00, val_acc:0.00, unlabeled_acc:0.00, shared_acc:0.00
                        Epoch: 7, labeled_acc:0.98, val_acc:0.98, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 8, labeled_acc:0.00, val_acc:0.00, unlabeled_acc:0.00, shared_acc:0.00
                        Epoch: 9, labeled_acc:0.98, val_acc:0.98, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 10, labeled_acc:0.00, val_acc:0.00, unlabeled_acc:0.00, shared_acc:0.00
                        Epoch: 11, labeled_acc:0.98, val_acc:0.98, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 12, labeled_acc:0.00, val_acc:0.00, unlabeled_acc:0.00, shared_acc:0.00
                        Epoch: 13, labeled_acc:0.98, val_acc:0.98, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 14, labeled_acc:0.00, val_acc:0.00, unlabeled_acc:0.00, shared_acc:0.00
                        Epoch: 15, labeled_acc:0.98, val_acc:0.97, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 16, labeled_acc:0.00, val_acc:0.00, unlabeled_acc:0.00, shared_acc:0.00
                        Epoch: 17, labeled_acc:0.98, val_acc:0.97, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 18, labeled_acc:0.00, val_acc:0.00, unlabeled_acc:0.00, shared_acc:0.00
                        Epoch: 19, labeled_acc:0.98, val_acc:0.98, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 20, labeled_acc:0.00, val_acc:0.00, unlabeled_acc:0.00, shared_acc:0.00
                        Epoch: 21, labeled_acc:0.98, val_acc:0.98, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 22, labeled_acc:0.00, val_acc:0.00, unlabeled_acc:0.00, shared_acc:0.00
                        Epoch: 23, labeled_acc:0.98, val_acc:0.98, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 24, labeled_acc:0.00, val_acc:0.00, unlabeled_acc:0.00, shared_acc:0.00
                        Epoch: 25, labeled_acc:0.98, val_acc:0.98, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 26, labeled_acc:0.00, val_acc:0.00, unlabeled_acc:0.00, shared_acc:0.00
                        Epoch: 27, labeled_acc:0.98, val_acc:0.98, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 28, labeled_acc:0.00, val_acc:0.00, unlabeled_acc:0.00, shared_acc:0.00
                        Epoch: 29, labeled_acc:0.98, val_acc:0.97, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 30, labeled_acc:0.95, val_acc:0.95, unlabeled_acc:0.96, shared_acc:0.95
                        Epoch: 31, labeled_acc:0.98, val_acc:0.98, unlabeled_acc:0.99, shared_acc:0.98
                        Epoch: 32, labeled_acc:0.00, val_acc:0.00, unlabeled_acc:0.00, shared_acc:0.00
                        Epoch: 33, labeled_acc:0.98, val_acc:0.98, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 34, labeled_acc:0.32, val_acc:0.34, unlabeled_acc:0.31, shared_acc:0.37
                        Epoch: 35, labeled_acc:0.98, val_acc:0.97, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 36, labeled_acc:0.00, val_acc:0.00, unlabeled_acc:0.00, shared_acc:0.00
                        Epoch: 37, labeled_acc:0.98, val_acc:0.98, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 38, labeled_acc:0.64, val_acc:0.63, unlabeled_acc:0.64, shared_acc:0.59
                        Epoch: 39, labeled_acc:0.98, val_acc:0.98, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 40, labeled_acc:0.02, val_acc:0.01, unlabeled_acc:0.02, shared_acc:0.06
                        Epoch: 41, labeled_acc:0.98, val_acc:0.98, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 42, labeled_acc:0.00, val_acc:0.00, unlabeled_acc:0.00, shared_acc:0.00
                        Epoch: 43, labeled_acc:0.98, val_acc:0.98, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 44, labeled_acc:0.98, val_acc:0.98, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 45, labeled_acc:0.04, val_acc:0.03, unlabeled_acc:0.04, shared_acc:0.08
                        Epoch: 46, labeled_acc:0.98, val_acc:0.98, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 47, labeled_acc:0.00, val_acc:0.00, unlabeled_acc:0.00, shared_acc:0.00
                        Epoch: 48, labeled_acc:0.98, val_acc:0.98, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 49, labeled_acc:0.02, val_acc:0.02, unlabeled_acc:0.02, shared_acc:0.07
                        Epoch: 50, labeled_acc:0.98, val_acc:0.98, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 51, labeled_acc:0.97, val_acc:0.97, unlabeled_acc:0.97, shared_acc:0.97
                        Epoch: 52, labeled_acc:0.98, val_acc:0.98, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 53, labeled_acc:0.96, val_acc:0.96, unlabeled_acc:0.97, shared_acc:0.97
                        Epoch: 54, labeled_acc:0.98, val_acc:0.98, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 55, labeled_acc:0.98, val_acc:0.98, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 56, labeled_acc:0.98, val_acc:0.98, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 57, labeled_acc:0.98, val_acc:0.98, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 58, labeled_acc:0.96, val_acc:0.95, unlabeled_acc:0.96, shared_acc:0.97
                        Epoch: 59, labeled_acc:0.98, val_acc:0.98, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 60, labeled_acc:0.96, val_acc:0.96, unlabeled_acc:0.97, shared_acc:0.97
                        Epoch: 61, labeled_acc:0.97, val_acc:0.98, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 62, labeled_acc:0.98, val_acc:0.98, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 63, labeled_acc:0.97, val_acc:0.98, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 64, labeled_acc:0.98, val_acc:0.98, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 65, labeled_acc:0.98, val_acc:0.98, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 66, labeled_acc:0.98, val_acc:0.98, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 67, labeled_acc:0.98, val_acc:0.98, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 68, labeled_acc:0.98, val_acc:0.98, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 69, labeled_acc:0.98, val_acc:0.98, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 70, labeled_acc:0.98, val_acc:0.98, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 71, labeled_acc:0.98, val_acc:0.97, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 72, labeled_acc:0.98, val_acc:0.98, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 73, labeled_acc:0.98, val_acc:0.98, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 74, labeled_acc:0.98, val_acc:0.98, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 75, labeled_acc:0.98, val_acc:0.98, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 76, labeled_acc:0.98, val_acc:0.98, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 77, labeled_acc:0.98, val_acc:0.98, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 78, labeled_acc:0.98, val_acc:0.98, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 79, labeled_acc:0.98, val_acc:0.98, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 80, labeled_acc:0.98, val_acc:0.99, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 81, labeled_acc:0.98, val_acc:0.98, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 82, labeled_acc:1.00, val_acc:0.99, unlabeled_acc:0.99, shared_acc:0.86
                        Epoch: 83, labeled_acc:0.99, val_acc:0.99, unlabeled_acc:0.99, shared_acc:0.98
                        Epoch: 84, labeled_acc:0.99, val_acc:0.98, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 85, labeled_acc:0.63, val_acc:0.62, unlabeled_acc:0.62, shared_acc:0.81
                        Epoch: 86, labeled_acc:0.98, val_acc:0.98, unlabeled_acc:0.97, shared_acc:0.98
                        Epoch: 87, labeled_acc:0.98, val_acc:0.99, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 88, labeled_acc:0.98, val_acc:0.99, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 89, labeled_acc:0.98, val_acc:0.99, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 90, labeled_acc:0.98, val_acc:0.98, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 91, labeled_acc:0.98, val_acc:0.99, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 92, labeled_acc:0.98, val_acc:0.98, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 93, labeled_acc:0.98, val_acc:0.99, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 94, labeled_acc:0.98, val_acc:0.98, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 95, labeled_acc:0.98, val_acc:0.98, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 96, labeled_acc:0.98, val_acc:0.98, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 97, labeled_acc:0.98, val_acc:0.98, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 98, labeled_acc:0.98, val_acc:0.98, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 99, labeled_acc:0.98, val_acc:0.98, unlabeled_acc:0.98, shared_acc:0.98
                       """,
                   'krum':
                       """
                        Epoch: 0, labeled_acc:0.13, val_acc:0.12, unlabeled_acc:0.12, shared_acc:0.14
                        Epoch: 1, labeled_acc:0.00, val_acc:0.00, unlabeled_acc:0.01, shared_acc:0.01
                        Epoch: 2, labeled_acc:0.92, val_acc:0.92, unlabeled_acc:0.93, shared_acc:0.91
                        Epoch: 3, labeled_acc:0.00, val_acc:0.00, unlabeled_acc:0.00, shared_acc:0.00
                        Epoch: 4, labeled_acc:0.97, val_acc:0.96, unlabeled_acc:0.97, shared_acc:0.97
                        Epoch: 5, labeled_acc:0.00, val_acc:0.00, unlabeled_acc:0.00, shared_acc:0.00
                        Epoch: 6, labeled_acc:0.97, val_acc:0.97, unlabeled_acc:0.98, shared_acc:0.97
                        Epoch: 7, labeled_acc:0.00, val_acc:0.00, unlabeled_acc:0.00, shared_acc:0.00
                        Epoch: 8, labeled_acc:0.98, val_acc:0.97, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 9, labeled_acc:0.00, val_acc:0.00, unlabeled_acc:0.00, shared_acc:0.00
                        Epoch: 10, labeled_acc:0.98, val_acc:0.97, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 11, labeled_acc:0.00, val_acc:0.00, unlabeled_acc:0.00, shared_acc:0.00
                        Epoch: 12, labeled_acc:0.98, val_acc:0.97, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 13, labeled_acc:0.00, val_acc:0.00, unlabeled_acc:0.00, shared_acc:0.00
                        Epoch: 14, labeled_acc:0.98, val_acc:0.97, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 15, labeled_acc:0.00, val_acc:0.00, unlabeled_acc:0.00, shared_acc:0.00
                        Epoch: 16, labeled_acc:0.98, val_acc:0.97, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 17, labeled_acc:0.00, val_acc:0.00, unlabeled_acc:0.00, shared_acc:0.00
                        Epoch: 18, labeled_acc:0.98, val_acc:0.97, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 19, labeled_acc:0.00, val_acc:0.00, unlabeled_acc:0.00, shared_acc:0.00
                        Epoch: 20, labeled_acc:0.98, val_acc:0.97, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 21, labeled_acc:0.00, val_acc:0.00, unlabeled_acc:0.00, shared_acc:0.00
                        Epoch: 22, labeled_acc:0.98, val_acc:0.97, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 23, labeled_acc:0.00, val_acc:0.00, unlabeled_acc:0.00, shared_acc:0.00
                        Epoch: 24, labeled_acc:0.98, val_acc:0.98, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 25, labeled_acc:0.00, val_acc:0.00, unlabeled_acc:0.00, shared_acc:0.00
                        Epoch: 26, labeled_acc:0.98, val_acc:0.98, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 27, labeled_acc:0.00, val_acc:0.00, unlabeled_acc:0.00, shared_acc:0.00
                        Epoch: 28, labeled_acc:0.98, val_acc:0.98, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 29, labeled_acc:0.00, val_acc:0.00, unlabeled_acc:0.00, shared_acc:0.00
                        Epoch: 30, labeled_acc:0.98, val_acc:0.98, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 31, labeled_acc:0.00, val_acc:0.00, unlabeled_acc:0.00, shared_acc:0.00
                        Epoch: 32, labeled_acc:0.98, val_acc:0.98, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 33, labeled_acc:0.00, val_acc:0.00, unlabeled_acc:0.00, shared_acc:0.00
                        Epoch: 34, labeled_acc:0.98, val_acc:0.98, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 35, labeled_acc:0.00, val_acc:0.00, unlabeled_acc:0.00, shared_acc:0.00
                        Epoch: 36, labeled_acc:0.98, val_acc:0.98, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 37, labeled_acc:0.00, val_acc:0.00, unlabeled_acc:0.00, shared_acc:0.00
                        Epoch: 38, labeled_acc:0.98, val_acc:0.98, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 39, labeled_acc:0.00, val_acc:0.00, unlabeled_acc:0.00, shared_acc:0.00
                        Epoch: 40, labeled_acc:0.98, val_acc:0.98, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 41, labeled_acc:0.00, val_acc:0.00, unlabeled_acc:0.00, shared_acc:0.00
                        Epoch: 42, labeled_acc:0.98, val_acc:0.98, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 43, labeled_acc:0.00, val_acc:0.00, unlabeled_acc:0.00, shared_acc:0.00
                        Epoch: 44, labeled_acc:0.98, val_acc:0.98, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 45, labeled_acc:0.00, val_acc:0.00, unlabeled_acc:0.00, shared_acc:0.00
                        Epoch: 46, labeled_acc:0.98, val_acc:0.98, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 47, labeled_acc:0.00, val_acc:0.00, unlabeled_acc:0.00, shared_acc:0.00
                        Epoch: 48, labeled_acc:0.98, val_acc:0.98, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 49, labeled_acc:0.00, val_acc:0.00, unlabeled_acc:0.00, shared_acc:0.00
                        Epoch: 50, labeled_acc:0.98, val_acc:0.98, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 51, labeled_acc:0.00, val_acc:0.00, unlabeled_acc:0.00, shared_acc:0.00
                        Epoch: 52, labeled_acc:0.98, val_acc:0.98, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 53, labeled_acc:0.00, val_acc:0.00, unlabeled_acc:0.00, shared_acc:0.00
                        Epoch: 54, labeled_acc:0.98, val_acc:0.98, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 55, labeled_acc:0.00, val_acc:0.00, unlabeled_acc:0.00, shared_acc:0.00
                        Epoch: 56, labeled_acc:0.98, val_acc:0.98, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 57, labeled_acc:0.00, val_acc:0.00, unlabeled_acc:0.00, shared_acc:0.00
                        Epoch: 58, labeled_acc:0.98, val_acc:0.98, unlabeled_acc:0.98, shared_acc:0.98
                        Epoch: 59, labeled_acc:0.00, val_acc:0.00, unlabeled_acc:0.00, shared_acc:0.00
                        Epoch: 60, labeled_acc:0.98, val_acc:0.98, unlabeled_acc:0.98, shared_acc:0.97
                        Epoch: 61, labeled_acc:0.00, val_acc:0.00, unlabeled_acc:0.00, shared_acc:0.01
                        Epoch: 62, labeled_acc:0.98, val_acc:0.98, unlabeled_acc:0.98, shared_acc:0.97
                        Epoch: 63, labeled_acc:0.00, val_acc:0.00, unlabeled_acc:0.00, shared_acc:0.01
                        Epoch: 64, labeled_acc:0.98, val_acc:0.98, unlabeled_acc:0.98, shared_acc:0.97
                        Epoch: 65, labeled_acc:0.00, val_acc:0.00, unlabeled_acc:0.01, shared_acc:0.01
                        Epoch: 66, labeled_acc:0.97, val_acc:0.98, unlabeled_acc:0.98, shared_acc:0.97
                        Epoch: 67, labeled_acc:0.00, val_acc:0.00, unlabeled_acc:0.01, shared_acc:0.01
                        Epoch: 68, labeled_acc:0.97, val_acc:0.98, unlabeled_acc:0.98, shared_acc:0.97
                        Epoch: 69, labeled_acc:0.00, val_acc:0.00, unlabeled_acc:0.01, shared_acc:0.01
                        Epoch: 70, labeled_acc:0.97, val_acc:0.98, unlabeled_acc:0.98, shared_acc:0.97
                        Epoch: 71, labeled_acc:0.00, val_acc:0.00, unlabeled_acc:0.01, shared_acc:0.01
                        Epoch: 72, labeled_acc:0.97, val_acc:0.98, unlabeled_acc:0.98, shared_acc:0.97
                        Epoch: 73, labeled_acc:0.00, val_acc:0.00, unlabeled_acc:0.01, shared_acc:0.01
                        Epoch: 74, labeled_acc:0.97, val_acc:0.97, unlabeled_acc:0.98, shared_acc:0.97
                        Epoch: 75, labeled_acc:0.00, val_acc:0.00, unlabeled_acc:0.01, shared_acc:0.01
                        Epoch: 76, labeled_acc:0.97, val_acc:0.97, unlabeled_acc:0.98, shared_acc:0.97
                        Epoch: 77, labeled_acc:0.00, val_acc:0.00, unlabeled_acc:0.01, shared_acc:0.01
                        Epoch: 78, labeled_acc:0.97, val_acc:0.97, unlabeled_acc:0.98, shared_acc:0.97
                        Epoch: 79, labeled_acc:0.00, val_acc:0.00, unlabeled_acc:0.01, shared_acc:0.01
                        Epoch: 80, labeled_acc:0.97, val_acc:0.97, unlabeled_acc:0.98, shared_acc:0.97
                        Epoch: 81, labeled_acc:0.00, val_acc:0.00, unlabeled_acc:0.01, shared_acc:0.01
                        Epoch: 82, labeled_acc:0.97, val_acc:0.98, unlabeled_acc:0.97, shared_acc:0.97
                        Epoch: 83, labeled_acc:0.00, val_acc:0.00, unlabeled_acc:0.00, shared_acc:0.01
                        Epoch: 84, labeled_acc:0.97, val_acc:0.98, unlabeled_acc:0.97, shared_acc:0.97
                        Epoch: 85, labeled_acc:0.00, val_acc:0.00, unlabeled_acc:0.01, shared_acc:0.01
                        Epoch: 86, labeled_acc:0.97, val_acc:0.97, unlabeled_acc:0.97, shared_acc:0.97
                        Epoch: 87, labeled_acc:0.00, val_acc:0.00, unlabeled_acc:0.00, shared_acc:0.01
                        Epoch: 88, labeled_acc:0.97, val_acc:0.97, unlabeled_acc:0.97, shared_acc:0.97
                        Epoch: 89, labeled_acc:0.00, val_acc:0.00, unlabeled_acc:0.00, shared_acc:0.01
                        Epoch: 90, labeled_acc:0.97, val_acc:0.97, unlabeled_acc:0.97, shared_acc:0.97
                        Epoch: 91, labeled_acc:0.00, val_acc:0.00, unlabeled_acc:0.00, shared_acc:0.01
                        Epoch: 92, labeled_acc:0.97, val_acc:0.97, unlabeled_acc:0.97, shared_acc:0.97
                        Epoch: 93, labeled_acc:0.00, val_acc:0.00, unlabeled_acc:0.00, shared_acc:0.00
                        Epoch: 94, labeled_acc:0.97, val_acc:0.97, unlabeled_acc:0.97, shared_acc:0.97
                        Epoch: 95, labeled_acc:0.00, val_acc:0.00, unlabeled_acc:0.00, shared_acc:0.01
                        Epoch: 96, labeled_acc:0.97, val_acc:0.97, unlabeled_acc:0.97, shared_acc:0.97
                        Epoch: 97, labeled_acc:0.01, val_acc:0.00, unlabeled_acc:0.00, shared_acc:0.01
                        Epoch: 98, labeled_acc:0.97, val_acc:0.97, unlabeled_acc:0.97, shared_acc:0.97
                        Epoch: 99, labeled_acc:0.01, val_acc:0.00, unlabeled_acc:0.01, shared_acc:0.01
                       """,
                   'median':
                       """
                        Epoch: 0, labeled_acc:0.13, val_acc:0.12, unlabeled_acc:0.12, shared_acc:0.14
                        Epoch: 1, labeled_acc:0.77, val_acc:0.75, unlabeled_acc:0.76, shared_acc:0.48
                        Epoch: 2, labeled_acc:0.76, val_acc:0.77, unlabeled_acc:0.77, shared_acc:0.71
                        Epoch: 3, labeled_acc:0.58, val_acc:0.59, unlabeled_acc:0.57, shared_acc:0.66
                        Epoch: 4, labeled_acc:0.77, val_acc:0.73, unlabeled_acc:0.77, shared_acc:0.63
                        Epoch: 5, labeled_acc:0.25, val_acc:0.27, unlabeled_acc:0.25, shared_acc:0.45
                        Epoch: 6, labeled_acc:0.78, val_acc:0.74, unlabeled_acc:0.78, shared_acc:0.60
                        Epoch: 7, labeled_acc:0.21, val_acc:0.24, unlabeled_acc:0.21, shared_acc:0.43
                        Epoch: 8, labeled_acc:0.79, val_acc:0.75, unlabeled_acc:0.79, shared_acc:0.60
                        Epoch: 9, labeled_acc:0.21, val_acc:0.23, unlabeled_acc:0.20, shared_acc:0.40
                        Epoch: 10, labeled_acc:0.78, val_acc:0.75, unlabeled_acc:0.79, shared_acc:0.61
                        Epoch: 11, labeled_acc:0.21, val_acc:0.23, unlabeled_acc:0.21, shared_acc:0.39
                        Epoch: 12, labeled_acc:0.78, val_acc:0.74, unlabeled_acc:0.78, shared_acc:0.60
                        Epoch: 13, labeled_acc:0.24, val_acc:0.27, unlabeled_acc:0.23, shared_acc:0.41
                        Epoch: 14, labeled_acc:0.76, val_acc:0.73, unlabeled_acc:0.77, shared_acc:0.59
                        Epoch: 15, labeled_acc:0.34, val_acc:0.37, unlabeled_acc:0.33, shared_acc:0.47
                        Epoch: 16, labeled_acc:0.68, val_acc:0.66, unlabeled_acc:0.68, shared_acc:0.54
                        Epoch: 17, labeled_acc:0.39, val_acc:0.41, unlabeled_acc:0.38, shared_acc:0.51
                        Epoch: 18, labeled_acc:0.61, val_acc:0.59, unlabeled_acc:0.62, shared_acc:0.51
                        Epoch: 19, labeled_acc:0.41, val_acc:0.42, unlabeled_acc:0.39, shared_acc:0.53
                        Epoch: 20, labeled_acc:0.61, val_acc:0.59, unlabeled_acc:0.62, shared_acc:0.51
                        Epoch: 21, labeled_acc:0.40, val_acc:0.42, unlabeled_acc:0.39, shared_acc:0.55
                        Epoch: 22, labeled_acc:0.61, val_acc:0.59, unlabeled_acc:0.62, shared_acc:0.48
                        Epoch: 23, labeled_acc:0.40, val_acc:0.42, unlabeled_acc:0.39, shared_acc:0.57
                        Epoch: 24, labeled_acc:0.62, val_acc:0.60, unlabeled_acc:0.63, shared_acc:0.45
                        Epoch: 25, labeled_acc:0.41, val_acc:0.42, unlabeled_acc:0.39, shared_acc:0.58
                        Epoch: 26, labeled_acc:0.62, val_acc:0.60, unlabeled_acc:0.62, shared_acc:0.44
                        Epoch: 27, labeled_acc:0.41, val_acc:0.42, unlabeled_acc:0.39, shared_acc:0.58
                        Epoch: 28, labeled_acc:0.62, val_acc:0.61, unlabeled_acc:0.63, shared_acc:0.44
                        Epoch: 29, labeled_acc:0.41, val_acc:0.42, unlabeled_acc:0.40, shared_acc:0.59
                        Epoch: 30, labeled_acc:0.63, val_acc:0.62, unlabeled_acc:0.63, shared_acc:0.45
                        Epoch: 31, labeled_acc:0.41, val_acc:0.42, unlabeled_acc:0.39, shared_acc:0.58
                        Epoch: 32, labeled_acc:0.64, val_acc:0.63, unlabeled_acc:0.65, shared_acc:0.46
                        Epoch: 33, labeled_acc:0.41, val_acc:0.42, unlabeled_acc:0.39, shared_acc:0.58
                        Epoch: 34, labeled_acc:0.65, val_acc:0.64, unlabeled_acc:0.66, shared_acc:0.50
                        Epoch: 35, labeled_acc:0.41, val_acc:0.43, unlabeled_acc:0.39, shared_acc:0.56
                        Epoch: 36, labeled_acc:0.68, val_acc:0.67, unlabeled_acc:0.68, shared_acc:0.55
                        Epoch: 37, labeled_acc:0.41, val_acc:0.43, unlabeled_acc:0.39, shared_acc:0.50
                        Epoch: 38, labeled_acc:0.72, val_acc:0.69, unlabeled_acc:0.71, shared_acc:0.57
                        Epoch: 39, labeled_acc:0.38, val_acc:0.39, unlabeled_acc:0.36, shared_acc:0.49
                        Epoch: 40, labeled_acc:0.76, val_acc:0.72, unlabeled_acc:0.76, shared_acc:0.60
                        Epoch: 41, labeled_acc:0.33, val_acc:0.33, unlabeled_acc:0.32, shared_acc:0.48
                        Epoch: 42, labeled_acc:0.61, val_acc:0.58, unlabeled_acc:0.61, shared_acc:0.53
                        Epoch: 43, labeled_acc:0.39, val_acc:0.40, unlabeled_acc:0.38, shared_acc:0.49
                        Epoch: 44, labeled_acc:0.61, val_acc:0.58, unlabeled_acc:0.60, shared_acc:0.52
                        Epoch: 45, labeled_acc:0.39, val_acc:0.42, unlabeled_acc:0.39, shared_acc:0.47
                        Epoch: 46, labeled_acc:0.62, val_acc:0.58, unlabeled_acc:0.61, shared_acc:0.53
                        Epoch: 47, labeled_acc:0.39, val_acc:0.42, unlabeled_acc:0.39, shared_acc:0.48
                        Epoch: 48, labeled_acc:0.62, val_acc:0.58, unlabeled_acc:0.62, shared_acc:0.53
                        Epoch: 49, labeled_acc:0.40, val_acc:0.43, unlabeled_acc:0.39, shared_acc:0.48
                        Epoch: 50, labeled_acc:0.62, val_acc:0.59, unlabeled_acc:0.62, shared_acc:0.53
                        Epoch: 51, labeled_acc:0.40, val_acc:0.43, unlabeled_acc:0.40, shared_acc:0.48
                        Epoch: 52, labeled_acc:0.62, val_acc:0.59, unlabeled_acc:0.62, shared_acc:0.53
                        Epoch: 53, labeled_acc:0.40, val_acc:0.43, unlabeled_acc:0.40, shared_acc:0.49
                        Epoch: 54, labeled_acc:0.63, val_acc:0.60, unlabeled_acc:0.62, shared_acc:0.54
                        Epoch: 55, labeled_acc:0.40, val_acc:0.43, unlabeled_acc:0.40, shared_acc:0.49
                        Epoch: 56, labeled_acc:0.62, val_acc:0.60, unlabeled_acc:0.62, shared_acc:0.54
                        Epoch: 57, labeled_acc:0.40, val_acc:0.43, unlabeled_acc:0.40, shared_acc:0.49
                        Epoch: 58, labeled_acc:0.62, val_acc:0.60, unlabeled_acc:0.62, shared_acc:0.54
                        Epoch: 59, labeled_acc:0.40, val_acc:0.43, unlabeled_acc:0.40, shared_acc:0.49
                        Epoch: 60, labeled_acc:0.62, val_acc:0.59, unlabeled_acc:0.62, shared_acc:0.54
                        Epoch: 61, labeled_acc:0.40, val_acc:0.43, unlabeled_acc:0.40, shared_acc:0.48
                        Epoch: 62, labeled_acc:0.62, val_acc:0.59, unlabeled_acc:0.62, shared_acc:0.54
                        Epoch: 63, labeled_acc:0.40, val_acc:0.43, unlabeled_acc:0.40, shared_acc:0.49
                        Epoch: 64, labeled_acc:0.63, val_acc:0.60, unlabeled_acc:0.62, shared_acc:0.54
                        Epoch: 65, labeled_acc:0.40, val_acc:0.43, unlabeled_acc:0.40, shared_acc:0.48
                        Epoch: 66, labeled_acc:0.62, val_acc:0.59, unlabeled_acc:0.62, shared_acc:0.54
                        Epoch: 67, labeled_acc:0.41, val_acc:0.43, unlabeled_acc:0.40, shared_acc:0.48
                        Epoch: 68, labeled_acc:0.62, val_acc:0.60, unlabeled_acc:0.62, shared_acc:0.54
                        Epoch: 69, labeled_acc:0.40, val_acc:0.43, unlabeled_acc:0.39, shared_acc:0.48
                        Epoch: 70, labeled_acc:0.62, val_acc:0.60, unlabeled_acc:0.62, shared_acc:0.54
                        Epoch: 71, labeled_acc:0.40, val_acc:0.43, unlabeled_acc:0.40, shared_acc:0.48
                        Epoch: 72, labeled_acc:0.62, val_acc:0.59, unlabeled_acc:0.62, shared_acc:0.53
                        Epoch: 73, labeled_acc:0.40, val_acc:0.42, unlabeled_acc:0.39, shared_acc:0.48
                        Epoch: 74, labeled_acc:0.62, val_acc:0.59, unlabeled_acc:0.62, shared_acc:0.53
                        Epoch: 75, labeled_acc:0.40, val_acc:0.42, unlabeled_acc:0.39, shared_acc:0.48
                        Epoch: 76, labeled_acc:0.62, val_acc:0.59, unlabeled_acc:0.62, shared_acc:0.54
                        Epoch: 77, labeled_acc:0.40, val_acc:0.43, unlabeled_acc:0.40, shared_acc:0.48
                        Epoch: 78, labeled_acc:0.62, val_acc:0.59, unlabeled_acc:0.62, shared_acc:0.53
                        Epoch: 79, labeled_acc:0.39, val_acc:0.42, unlabeled_acc:0.39, shared_acc:0.48
                        Epoch: 80, labeled_acc:0.61, val_acc:0.59, unlabeled_acc:0.62, shared_acc:0.54
                        Epoch: 81, labeled_acc:0.40, val_acc:0.43, unlabeled_acc:0.39, shared_acc:0.47
                        Epoch: 82, labeled_acc:0.61, val_acc:0.59, unlabeled_acc:0.61, shared_acc:0.53
                        Epoch: 83, labeled_acc:0.40, val_acc:0.43, unlabeled_acc:0.39, shared_acc:0.48
                        Epoch: 84, labeled_acc:0.62, val_acc:0.59, unlabeled_acc:0.62, shared_acc:0.53
                        Epoch: 85, labeled_acc:0.39, val_acc:0.42, unlabeled_acc:0.39, shared_acc:0.48
                        Epoch: 86, labeled_acc:0.61, val_acc:0.58, unlabeled_acc:0.61, shared_acc:0.54
                        Epoch: 87, labeled_acc:0.40, val_acc:0.43, unlabeled_acc:0.40, shared_acc:0.48
                        Epoch: 88, labeled_acc:0.61, val_acc:0.59, unlabeled_acc:0.61, shared_acc:0.53
                        Epoch: 89, labeled_acc:0.39, val_acc:0.42, unlabeled_acc:0.39, shared_acc:0.48
                        Epoch: 90, labeled_acc:0.61, val_acc:0.59, unlabeled_acc:0.62, shared_acc:0.54
                        Epoch: 91, labeled_acc:0.39, val_acc:0.42, unlabeled_acc:0.39, shared_acc:0.47
                        Epoch: 92, labeled_acc:0.61, val_acc:0.59, unlabeled_acc:0.61, shared_acc:0.53
                        Epoch: 93, labeled_acc:0.40, val_acc:0.43, unlabeled_acc:0.39, shared_acc:0.48
                        Epoch: 94, labeled_acc:0.61, val_acc:0.59, unlabeled_acc:0.61, shared_acc:0.52
                        Epoch: 95, labeled_acc:0.39, val_acc:0.42, unlabeled_acc:0.39, shared_acc:0.48
                        Epoch: 96, labeled_acc:0.61, val_acc:0.59, unlabeled_acc:0.61, shared_acc:0.54
                        Epoch: 97, labeled_acc:0.39, val_acc:0.42, unlabeled_acc:0.39, shared_acc:0.47
                        Epoch: 98, labeled_acc:0.60, val_acc:0.58, unlabeled_acc:0.60, shared_acc:0.52
                        Epoch: 99, labeled_acc:0.39, val_acc:0.42, unlabeled_acc:0.39, shared_acc:0.48
                       """,
                   'mean':
                       """
                        Epoch: 0, labeled_acc:0.13, val_acc:0.12, unlabeled_acc:0.12, shared_acc:0.14
                        Epoch: 1, labeled_acc:0.65, val_acc:0.64, unlabeled_acc:0.64, shared_acc:0.48
                        Epoch: 2, labeled_acc:0.75, val_acc:0.74, unlabeled_acc:0.74, shared_acc:0.62
                        Epoch: 3, labeled_acc:0.91, val_acc:0.92, unlabeled_acc:0.90, shared_acc:0.85
                        Epoch: 4, labeled_acc:0.93, val_acc:0.93, unlabeled_acc:0.93, shared_acc:0.67
                        Epoch: 5, labeled_acc:0.89, val_acc:0.86, unlabeled_acc:0.88, shared_acc:0.78
                        Epoch: 6, labeled_acc:0.93, val_acc:0.92, unlabeled_acc:0.93, shared_acc:0.87
                        Epoch: 7, labeled_acc:0.94, val_acc:0.93, unlabeled_acc:0.93, shared_acc:0.93
                        Epoch: 8, labeled_acc:0.96, val_acc:0.96, unlabeled_acc:0.96, shared_acc:0.93
                        Epoch: 9, labeled_acc:0.92, val_acc:0.90, unlabeled_acc:0.91, shared_acc:0.89
                        Epoch: 10, labeled_acc:0.90, val_acc:0.89, unlabeled_acc:0.89, shared_acc:0.87
                        Epoch: 11, labeled_acc:0.76, val_acc:0.73, unlabeled_acc:0.76, shared_acc:0.79
                        Epoch: 12, labeled_acc:0.72, val_acc:0.73, unlabeled_acc:0.72, shared_acc:0.73
                        Epoch: 13, labeled_acc:0.60, val_acc:0.55, unlabeled_acc:0.59, shared_acc:0.64
                        Epoch: 14, labeled_acc:0.61, val_acc:0.64, unlabeled_acc:0.61, shared_acc:0.61
                        Epoch: 15, labeled_acc:0.51, val_acc:0.46, unlabeled_acc:0.51, shared_acc:0.57
                        Epoch: 16, labeled_acc:0.60, val_acc:0.63, unlabeled_acc:0.60, shared_acc:0.57
                        Epoch: 17, labeled_acc:0.50, val_acc:0.45, unlabeled_acc:0.49, shared_acc:0.55
                        Epoch: 18, labeled_acc:0.58, val_acc:0.61, unlabeled_acc:0.58, shared_acc:0.55
                        Epoch: 19, labeled_acc:0.50, val_acc:0.45, unlabeled_acc:0.49, shared_acc:0.54
                        Epoch: 20, labeled_acc:0.58, val_acc:0.60, unlabeled_acc:0.57, shared_acc:0.54
                        Epoch: 21, labeled_acc:0.49, val_acc:0.45, unlabeled_acc:0.48, shared_acc:0.54
                        Epoch: 22, labeled_acc:0.57, val_acc:0.59, unlabeled_acc:0.56, shared_acc:0.53
                        Epoch: 23, labeled_acc:0.49, val_acc:0.45, unlabeled_acc:0.48, shared_acc:0.53
                        Epoch: 24, labeled_acc:0.56, val_acc:0.58, unlabeled_acc:0.55, shared_acc:0.52
                        Epoch: 25, labeled_acc:0.48, val_acc:0.44, unlabeled_acc:0.47, shared_acc:0.53
                        Epoch: 26, labeled_acc:0.56, val_acc:0.57, unlabeled_acc:0.54, shared_acc:0.51
                        Epoch: 27, labeled_acc:0.48, val_acc:0.44, unlabeled_acc:0.47, shared_acc:0.53
                        Epoch: 28, labeled_acc:0.55, val_acc:0.56, unlabeled_acc:0.54, shared_acc:0.50
                        Epoch: 29, labeled_acc:0.48, val_acc:0.44, unlabeled_acc:0.47, shared_acc:0.52
                        Epoch: 30, labeled_acc:0.55, val_acc:0.55, unlabeled_acc:0.53, shared_acc:0.50
                        Epoch: 31, labeled_acc:0.48, val_acc:0.43, unlabeled_acc:0.46, shared_acc:0.52
                        Epoch: 32, labeled_acc:0.54, val_acc:0.55, unlabeled_acc:0.52, shared_acc:0.49
                        Epoch: 33, labeled_acc:0.47, val_acc:0.42, unlabeled_acc:0.46, shared_acc:0.52
                        Epoch: 34, labeled_acc:0.54, val_acc:0.54, unlabeled_acc:0.52, shared_acc:0.49
                        Epoch: 35, labeled_acc:0.47, val_acc:0.41, unlabeled_acc:0.45, shared_acc:0.52
                        Epoch: 36, labeled_acc:0.53, val_acc:0.54, unlabeled_acc:0.51, shared_acc:0.49
                        Epoch: 37, labeled_acc:0.46, val_acc:0.41, unlabeled_acc:0.44, shared_acc:0.51
                        Epoch: 38, labeled_acc:0.53, val_acc:0.54, unlabeled_acc:0.51, shared_acc:0.49
                        Epoch: 39, labeled_acc:0.46, val_acc:0.40, unlabeled_acc:0.44, shared_acc:0.51
                        Epoch: 40, labeled_acc:0.53, val_acc:0.54, unlabeled_acc:0.51, shared_acc:0.48
                        Epoch: 41, labeled_acc:0.45, val_acc:0.39, unlabeled_acc:0.44, shared_acc:0.51
                        Epoch: 42, labeled_acc:0.53, val_acc:0.54, unlabeled_acc:0.51, shared_acc:0.48
                        Epoch: 43, labeled_acc:0.45, val_acc:0.39, unlabeled_acc:0.44, shared_acc:0.51
                        Epoch: 44, labeled_acc:0.53, val_acc:0.53, unlabeled_acc:0.51, shared_acc:0.48
                        Epoch: 45, labeled_acc:0.45, val_acc:0.39, unlabeled_acc:0.44, shared_acc:0.51
                        Epoch: 46, labeled_acc:0.53, val_acc:0.52, unlabeled_acc:0.51, shared_acc:0.48
                        Epoch: 47, labeled_acc:0.45, val_acc:0.39, unlabeled_acc:0.45, shared_acc:0.52
                        Epoch: 48, labeled_acc:0.52, val_acc:0.52, unlabeled_acc:0.49, shared_acc:0.48
                        Epoch: 49, labeled_acc:0.46, val_acc:0.39, unlabeled_acc:0.44, shared_acc:0.52
                        Epoch: 50, labeled_acc:0.51, val_acc:0.51, unlabeled_acc:0.48, shared_acc:0.47
                        Epoch: 51, labeled_acc:0.46, val_acc:0.40, unlabeled_acc:0.45, shared_acc:0.52
                        Epoch: 52, labeled_acc:0.48, val_acc:0.49, unlabeled_acc:0.46, shared_acc:0.46
                        Epoch: 53, labeled_acc:0.45, val_acc:0.40, unlabeled_acc:0.43, shared_acc:0.51
                        Epoch: 54, labeled_acc:0.47, val_acc:0.47, unlabeled_acc:0.45, shared_acc:0.46
                        Epoch: 55, labeled_acc:0.46, val_acc:0.40, unlabeled_acc:0.44, shared_acc:0.50
                        Epoch: 56, labeled_acc:0.48, val_acc:0.48, unlabeled_acc:0.46, shared_acc:0.48
                        Epoch: 57, labeled_acc:0.45, val_acc:0.40, unlabeled_acc:0.44, shared_acc:0.49
                        Epoch: 58, labeled_acc:0.51, val_acc:0.50, unlabeled_acc:0.49, shared_acc:0.51
                        Epoch: 59, labeled_acc:0.43, val_acc:0.38, unlabeled_acc:0.42, shared_acc:0.46
                        Epoch: 60, labeled_acc:0.57, val_acc:0.56, unlabeled_acc:0.55, shared_acc:0.55
                        Epoch: 61, labeled_acc:0.39, val_acc:0.35, unlabeled_acc:0.38, shared_acc:0.44
                        Epoch: 62, labeled_acc:0.58, val_acc:0.57, unlabeled_acc:0.56, shared_acc:0.54
                        Epoch: 63, labeled_acc:0.39, val_acc:0.35, unlabeled_acc:0.38, shared_acc:0.46
                        Epoch: 64, labeled_acc:0.58, val_acc:0.56, unlabeled_acc:0.57, shared_acc:0.53
                        Epoch: 65, labeled_acc:0.39, val_acc:0.35, unlabeled_acc:0.38, shared_acc:0.47
                        Epoch: 66, labeled_acc:0.60, val_acc:0.59, unlabeled_acc:0.59, shared_acc:0.53
                        Epoch: 67, labeled_acc:0.39, val_acc:0.35, unlabeled_acc:0.38, shared_acc:0.47
                        Epoch: 68, labeled_acc:0.61, val_acc:0.60, unlabeled_acc:0.59, shared_acc:0.54
                        Epoch: 69, labeled_acc:0.39, val_acc:0.35, unlabeled_acc:0.38, shared_acc:0.48
                        Epoch: 70, labeled_acc:0.61, val_acc:0.60, unlabeled_acc:0.60, shared_acc:0.54
                        Epoch: 71, labeled_acc:0.39, val_acc:0.35, unlabeled_acc:0.38, shared_acc:0.48
                        Epoch: 72, labeled_acc:0.61, val_acc:0.60, unlabeled_acc:0.60, shared_acc:0.54
                        Epoch: 73, labeled_acc:0.39, val_acc:0.35, unlabeled_acc:0.38, shared_acc:0.48
                        Epoch: 74, labeled_acc:0.62, val_acc:0.61, unlabeled_acc:0.61, shared_acc:0.54
                        Epoch: 75, labeled_acc:0.39, val_acc:0.35, unlabeled_acc:0.38, shared_acc:0.48
                        Epoch: 76, labeled_acc:0.62, val_acc:0.62, unlabeled_acc:0.61, shared_acc:0.54
                        Epoch: 77, labeled_acc:0.39, val_acc:0.35, unlabeled_acc:0.38, shared_acc:0.48
                        Epoch: 78, labeled_acc:0.62, val_acc:0.62, unlabeled_acc:0.61, shared_acc:0.54
                        Epoch: 79, labeled_acc:0.39, val_acc:0.35, unlabeled_acc:0.38, shared_acc:0.48
                        Epoch: 80, labeled_acc:0.62, val_acc:0.62, unlabeled_acc:0.61, shared_acc:0.54
                        Epoch: 81, labeled_acc:0.39, val_acc:0.36, unlabeled_acc:0.38, shared_acc:0.48
                        Epoch: 82, labeled_acc:0.62, val_acc:0.62, unlabeled_acc:0.62, shared_acc:0.54
                        Epoch: 83, labeled_acc:0.39, val_acc:0.36, unlabeled_acc:0.38, shared_acc:0.48
                        Epoch: 84, labeled_acc:0.63, val_acc:0.63, unlabeled_acc:0.62, shared_acc:0.54
                        Epoch: 85, labeled_acc:0.39, val_acc:0.36, unlabeled_acc:0.38, shared_acc:0.48
                        Epoch: 86, labeled_acc:0.63, val_acc:0.63, unlabeled_acc:0.62, shared_acc:0.55
                        Epoch: 87, labeled_acc:0.39, val_acc:0.36, unlabeled_acc:0.37, shared_acc:0.48
                        Epoch: 88, labeled_acc:0.63, val_acc:0.63, unlabeled_acc:0.62, shared_acc:0.55
                        Epoch: 89, labeled_acc:0.39, val_acc:0.36, unlabeled_acc:0.38, shared_acc:0.48
                        Epoch: 90, labeled_acc:0.63, val_acc:0.64, unlabeled_acc:0.63, shared_acc:0.55
                        Epoch: 91, labeled_acc:0.39, val_acc:0.35, unlabeled_acc:0.38, shared_acc:0.48
                        Epoch: 92, labeled_acc:0.63, val_acc:0.64, unlabeled_acc:0.62, shared_acc:0.55
                        Epoch: 93, labeled_acc:0.39, val_acc:0.36, unlabeled_acc:0.38, shared_acc:0.48
                        Epoch: 94, labeled_acc:0.63, val_acc:0.64, unlabeled_acc:0.63, shared_acc:0.55
                        Epoch: 95, labeled_acc:0.39, val_acc:0.36, unlabeled_acc:0.38, shared_acc:0.48
                        Epoch: 96, labeled_acc:0.64, val_acc:0.64, unlabeled_acc:0.63, shared_acc:0.55
                        Epoch: 97, labeled_acc:0.39, val_acc:0.36, unlabeled_acc:0.38, shared_acc:0.48
                        Epoch: 98, labeled_acc:0.64, val_acc:0.64, unlabeled_acc:0.63, shared_acc:0.55
                        Epoch: 99, labeled_acc:0.39, val_acc:0.35, unlabeled_acc:0.37, shared_acc:0.48
                       """
                   }

    # Extracting 'shared_acc' values from the strings
    shared_accs = {}
    for method, log in global_accs.items():
        # Regular expression to find 'shared_acc' values
        shared_accs[method] = [float(x) for x in re.findall(r"shared_acc:(0.\d+)", log)]

    # Display the shared_accs for each method
    print(shared_accs)

    aggregation_methods = list(global_accs.keys())
    makers = ['o', '+', 's', '*']
    for i in range(len(aggregation_methods)):
        agg_method = aggregation_methods[i]
        label = agg_method
        ys = shared_accs[agg_method]
        xs = range(len(ys))
        plt.plot(xs, ys, label=label, marker=makers[i])
    plt.xlabel('Server Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'Global CNN')
    plt.legend(fontsize=6.5, loc='lower right')

    # attacker_ratio = NUM_BYZANTINE_CLIENTS / (NUM_BENIGN_CLIENTS + NUM_BYZANTINE_CLIENTS)
    # title = (f'{model_type}_cnn' + '$_{' + f'{num_server_epoches}+1' + '}$' +
    #          f':{attacker_ratio:.2f}-{LABELING_RATE:.2f}')

    # Adjust layout to prevent overlap
    plt.tight_layout()
    # fig_file = (f'{IN_DIR}/{model_type}_{LABELING_RATE}_{AGGREGATION_METHOD}_'
    #             f'{SERVER_EPOCHS}_{NUM_BENIGN_CLIENTS}_{NUM_BYZANTINE_CLIENTS}_accuracy.png')
    fig_file = 'global_cnn.png'
    # os.makedirs(os.path.dirname(fig_file), exist_ok=True)
    plt.savefig(fig_file, dpi=300)
    plt.show()


if __name__ == '__main__':
    plot_robust_aggregation()
