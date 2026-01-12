import time, psutil, os, sys
from concurrent.futures import ThreadPoolExecutor

def task(n):
    return sum(range(n))

def run_thread_pool(num_tasks, n):
    with ThreadPoolExecutor() as executor:
        list(executor.map(task, [n] * num_tasks))

if __name__ == "__main__":
    num_tasks = int(sys.argv[1])
    n = int(sys.argv[2])

    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024*1024)

    start = time.perf_counter()
    run_thread_pool(num_tasks, n)
    end = time.perf_counter()

    mem_after = process.memory_info().rss / (1024*1024)
    elapsed = end - start
    throughput = num_tasks / elapsed

    print(f"[Thread Pool] {num_tasks} tasks")
    print(f"Time: {elapsed:.2f} s")
    print(f"Throughput: {throughput:.2f} tasks/s")
    print(f"Memory change: {mem_after - mem_before:.2f} MB")