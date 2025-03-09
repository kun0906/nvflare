import time
import numpy as np
import time
from datetime import datetime


def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_time_readable = datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')
        print(f"Function '{func.__name__} starts at {start_time_readable}..."
              )
        result = func(*args, **kwargs)
        end_time = time.time()
        end_time_readable = datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')
        elapsed_time = end_time - start_time
        print(
            f"Function '{func.__name__}' executed in {elapsed_time:.4f} seconds "
            f"(Start: {start_time_readable}, End: {end_time_readable})"
        )
        return result

    return wrapper


from numpy.random import dirichlet


def dirichlet_split(X, y, num_clients, alpha=0.5, random_state=42):
    """Splits dataset using Dirichlet distribution for non-IID allocation.

    alpha: > 0
        how class samples are divided among clients.
        Small alpha (e.g., 0.1) → Highly Non-IID
            Each client receives data dominated by a few classes.
            Some clients may not have samples from certain classes.

        Large alpha (e.g., 10) → More IID-like
            Each client receives a more balanced mix of all classes.
            The distribution approaches uniformity as alpha increases.

        alpha = 1 → Mildly Non-IID
            Classes are somewhat skewed, but each client still has a mix of multiple classes.

    """
    np.random.seed(random_state)  # Set random seed for reproducibility

    classes = np.unique(y)
    class_indices = {c: np.where(y == c)[0] for c in classes}
    X_splits, y_splits = [[] for _ in range(num_clients)], [[] for _ in range(num_clients)]
    print('\n\t: size, ' + ",  ".join(f'Client_{c}' for c in range(num_clients)))
    for c, indices in class_indices.items():
        np.random.shuffle(indices)
        proportions = dirichlet(alpha * np.ones(num_clients))
        proportions = (proportions * len(indices)).astype(int)

        # even split the left data to each client
        left = len(indices) - sum(proportions)
        left_per_each = left // num_clients
        if left_per_each > 0:
            proportions = [v + left_per_each for v in proportions]
        # Adjust the last client to ensure total sum matches
        proportions[-1] += len(indices) - sum(proportions)
        print(f'class {c}: ', sum(proportions), list(proportions), left, f', alpha: {alpha}')

        start = 0
        for client, num_samples in enumerate(proportions):
            X_splits[client].extend(X[indices[start:start + num_samples]])
            y_splits[client].extend(y[indices[start:start + num_samples]])
            start += num_samples

    return [np.array(X_s) for X_s in X_splits], [np.array(y_s) for y_s in y_splits]
