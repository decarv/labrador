import timeit
import itertools
import numpy as np

# Assuming ids is a list of 1 million ids
ids = list(range(1, 1000001))
workers = 7

def my_func(ids, workers):
    workers: int = min(len(ids), workers)
    first_ind: int = 0
    range_len: int = (len(ids) // workers) + 1
    id_ranges: list[tuple[int, int]] = []
    for i in range(workers):
        last_ind: int = range_len * (i + 1)
        if last_ind >= len(ids):
            last_ind = -1
        id_ranges.append((ids[first_ind], ids[last_ind]))
        first_ind = last_ind + 1

    return workers, id_ranges

def numpy_func(ids, workers):
    workers: int = min(len(ids), workers)
    id_ranges = [tuple(chunk[[0, -1]]) if len(chunk) > 0 else (None, None) for chunk in np.array_split(ids, workers)]
    return workers, id_ranges


def time_func(func, ids, workers):
    return timeit.timeit(lambda: func(ids, workers), number=10)


if __name__ == "__main__":
    print(f"my_func: {time_func(my_func, ids, workers)}")
    # print(f"iter_func: {time_func(iter_func, ids, workers)}")
    print(f"numpy_func: {time_func(numpy_func, ids, workers)}")

    print(f"my_func: {my_func(ids, workers)}")
    # print(f"iter_func: {iter_func(ids, workers)}")
    print(f"numpy_func: {numpy_func(ids, workers)}")
