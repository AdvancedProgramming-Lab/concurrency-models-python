import subprocess
import os

# Get current directory of controller.py
base_dir = os.path.dirname(__file__)

def run_experiment(script_name, num_tasks, n):
    script_path = os.path.join(base_dir, script_name)
    result = subprocess.run(
        ["python", "-u", script_path, str(num_tasks), str(n)],
        capture_output=True,
        text=True
    )
    if result.stdout.strip():
        print(result.stdout)
    if result.stderr.strip():
        print("Error:", result.stderr)

if __name__ == "__main__":
    for num in [10, 100, 1000]:
        print(f"\n=== {num} tasks ===")
        run_experiment("new_threads.py", num, 1000000)
        run_experiment("thread_pool.py", num, 1000000)