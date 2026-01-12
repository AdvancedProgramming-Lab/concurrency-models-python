import threading, time, psutil, os, sys

def task(n):
    return sum(range(n))

def run_new_threads(num_tasks, n):
    threads = []
    for _ in range(num_tasks):
        t = threading.Thread(target=task, args=(n,))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

if __name__ == "__main__":
    num_tasks = int(sys.argv[1])
    n = int(sys.argv[2])

    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024*1024)

    start = time.perf_counter()
    run_new_threads(num_tasks, n)
    end = time.perf_counter()

    mem_after = process.memory_info().rss / (1024*1024)
    elapsed = end - start
    throughput = num_tasks / elapsed

    print(f"[New Threads] {num_tasks} tasks")
    print(f"Time: {elapsed:.2f} s")
    print(f"Throughput: {throughput:.2f} tasks/s")
    print(f"Memory change: {mem_after - mem_before:.2f} MB")