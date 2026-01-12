import threading, time, psutil, os
from concurrent.futures import ThreadPoolExecutor

# -----------------------------
# Task definitions
# -----------------------------
def cpu_task(n):
    """CPU-bound task: sum numbers"""
    return sum(range(n)) #substituir por image processing (antes vs depois)/ factorial

def io_task(duration=1):
    """IO-bound task: sleep for given seconds"""
    time.sleep(duration) # crawler, sleep
    return duration 

# -----------------------------
# Approaches
# -----------------------------
def run_new_threads(num_tasks, func, *args):
    threads = []
    for _ in range(num_tasks):
        t = threading.Thread(target=func, args=args)
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

def run_thread_pool(num_tasks, func, *args, max_workers=None):
    with ThreadPoolExecutor(max_workers=max_workers) as executor: #processpoolexecutor
        list(executor.map(lambda _: func(*args), range(num_tasks)))

# -----------------------------
# Optimal pool size selector
# -----------------------------
def optimal_pool_size(workload="cpu"):
    cores = os.cpu_count()
    if workload == "cpu":
        return cores  # rule of thumb: cores
    elif workload == "io":
        return cores * 4  # rule of thumb: multiple of cores
    else:
        return cores

# -----------------------------
# Measurement helper
# -----------------------------
def measure(run_func, num_tasks, label, func, *args, **kwargs):
    process = psutil.Process(os.getpid())

    mem_before = process.memory_info().rss / (1024*1024)  # MB
    cpu_before = psutil.cpu_percent(interval=None)

    start = time.perf_counter()
    run_func(num_tasks, func, *args, **kwargs)
    end = time.perf_counter()

    mem_after = process.memory_info().rss / (1024*1024)  # MB
    cpu_after = psutil.cpu_percent(interval=None)

    elapsed = end - start
    throughput = num_tasks / elapsed
    mem_used = mem_after - mem_before
    cpu_used = cpu_after - cpu_before

    print(f"\n[{label}] {num_tasks} tasks")
    print(f"Time: {elapsed:.2f} s")
    print(f"Throughput: {throughput:.2f} tasks/s")
    print(f"Memory change: {mem_used:.2f} MB")
    print(f"CPU change: {cpu_used:.2f}%")

# -----------------------------
# Main experiment
# -----------------------------
if __name__ == "__main__":
    cores = os.cpu_count()
    print(f"Detected {cores} CPU cores")

    for num in [10, 100, 1000]:
        # CPU-bound tasks
        measure(run_new_threads, num, "New Threads (CPU)", cpu_task, 1000000)
        measure(run_thread_pool, num, f"Thread Pool (CPU, optimal={optimal_pool_size('cpu')})",
                cpu_task, 1000000, max_workers=optimal_pool_size("cpu"))

        # IO-bound tasks
        measure(run_new_threads, num, "New Threads (IO, sleep=0.1)", io_task, 0.1)
        measure(run_thread_pool, num, f"Thread Pool (IO, optimal={optimal_pool_size('io')})",
                io_task, 0.1, max_workers=optimal_pool_size("io"))