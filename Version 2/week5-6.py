from concurrent.futures import ThreadPoolExecutor
import time

def task(n):
    return sum(range(n))

def run_thread_pool(num_tasks, n):
    start = time.perf_counter()
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(task, [n]*num_tasks))
    end = time.perf_counter()
    print(f"{num_tasks} tasks took {end - start:.2f} seconds")

if __name__ == "__main__":
    run_thread_pool(10, 1000000)
    run_thread_pool(100, 1000000)
    run_thread_pool(1000, 1000000)