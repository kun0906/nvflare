import time

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
