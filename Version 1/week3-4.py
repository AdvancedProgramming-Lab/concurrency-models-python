import threading, time

def task(n):
    return sum(range(n))

def run_new_threads(num_tasks, n):
    threads = []
    start = time.perf_counter()
    for _ in range(num_tasks):
        t = threading.Thread(target=task, args=(n,))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    end = time.perf_counter()
    print(f"{num_tasks} tasks took {end - start:.2f} seconds")

if __name__ == "__main__":
    run_new_threads(10, 1000000)
    run_new_threads(100, 1000000)
    run_new_threads(1000, 1000000)