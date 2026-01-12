import threading
import time
import psutil
import os
import platform
import statistics
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
import requests
from bs4 import BeautifulSoup
from io import BytesIO
from typing import List, Dict, Any
from functools import partial

# =====================================================================
# SYSTEM INFORMATION COLLECTOR
# =====================================================================
class SystemInfo:
    """Collect detailed system information for comparison"""

    @staticmethod
    def collect():
        """Gather all relevant system information"""
        import cpuinfo

        # CPU information
        cpu_info = cpuinfo.get_cpu_info()

        # Memory information
        memory = psutil.virtual_memory()

        # Python version
        python_version = platform.python_version()

        return {
            "system_name": platform.node(),
            "os": f"{platform.system()} {platform.release()}",
            "python_version": python_version,
            "cpu": {
                "brand": cpu_info.get('brand_raw', 'Unknown'),
                "cores_physical": psutil.cpu_count(logical=False),
                "cores_logical": psutil.cpu_count(logical=True),
                "frequency_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else 0,
                "architecture": platform.machine(),
            },
            "memory": {
                "total_gb": round(memory.total / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2),
            },
            "test_timestamp": datetime.now().isoformat(),
        }

# =====================================================================
# WORKER WRAPPER
# =====================================================================
def worker_wrapper(task_data):
    """Wrapper function that can be pickled for ProcessPoolExecutor"""
    func, args = task_data
    return func(*args)

# =====================================================================
# TASK DEFINITIONS
# =====================================================================
def generate_test_image(size=(800, 600)):
    """Generate a colorful test image for processing"""
    img = Image.new('RGB', size, color=(135, 206, 235))
    draw = ImageDraw.Draw(img)

    # Sun
    draw.ellipse([650, 50, 750, 150], fill=(255, 255, 0))
    # Ground
    draw.rectangle([0, 400, 800, 600], fill=(34, 139, 34))
    # House
    draw.rectangle([200, 300, 400, 500], fill=(139, 69, 19))
    draw.polygon([(200, 300), (300, 200), (400, 300)], fill=(178, 34, 34))
    # Windows
    draw.rectangle([230, 350, 270, 400], fill=(173, 216, 230))
    draw.rectangle([330, 350, 370, 400], fill=(173, 216, 230))
    # Door
    draw.rectangle([280, 420, 320, 500], fill=(101, 67, 33))
    # Trees
    draw.rectangle([500, 380, 520, 500], fill=(101, 67, 33))
    draw.ellipse([460, 320, 560, 420], fill=(0, 100, 0))
    draw.rectangle([100, 380, 120, 500], fill=(101, 67, 33))
    draw.ellipse([60, 320, 160, 420], fill=(0, 128, 0))

    return img

def image_processing_task(image_data: bytes) -> bytes:
    """HEAVY CPU-intensive image processing"""
    img = Image.open(BytesIO(image_data))

    # Apply multiple heavy filters
    img = img.filter(ImageFilter.GaussianBlur(radius=10))
    img = img.filter(ImageFilter.GaussianBlur(radius=8))
    img = ImageEnhance.Contrast(img).enhance(1.8)
    img = ImageEnhance.Brightness(img).enhance(1.2)
    img = ImageEnhance.Color(img).enhance(1.5)
    img = ImageEnhance.Sharpness(img).enhance(2.0)
    img = img.filter(ImageFilter.FIND_EDGES)
    img = img.filter(ImageFilter.SMOOTH_MORE)
    img = img.filter(ImageFilter.SMOOTH)
    img = img.filter(ImageFilter.DETAIL)
    img = img.filter(ImageFilter.EDGE_ENHANCE)
    img = img.filter(ImageFilter.EDGE_ENHANCE_MORE)
    img = img.filter(ImageFilter.GaussianBlur(radius=5))

    output = BytesIO()
    img.save(output, format='PNG')
    return output.getvalue()

def factorial_task(n: int) -> int:
    """CPU-intensive: Calculate factorial iteratively"""
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def web_crawler_task(url: str, depth: int = 1) -> dict:
    """IO-bound web crawler"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Research Bot/1.0; Thread Pool Analysis)',
            'Accept': 'text/html,application/xhtml+xml',
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        title = soup.title.string if soup.title else "No title"
        links = [a.get('href') for a in soup.find_all('a', href=True)][:10]

        return {
            'url': url,
            'status': response.status_code,
            'title': title,
            'links_found': len(links),
            'content_length': len(response.text)
        }
    except Exception as e:
        return {'url': url, 'status': 'error', 'error': str(e)}

def sleep_task(duration: float = 0.1) -> float:
    """Simple IO-bound task: sleep"""
    time.sleep(duration)
    return duration

# =====================================================================
# EXECUTION APPROACHES
# =====================================================================
def run_new_threads(num_tasks: int, func, *args):
    """Create a new thread for each task"""
    threads = []
    for _ in range(num_tasks):
        t = threading.Thread(target=func, args=args)
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

def run_thread_pool(num_tasks: int, func, *args, max_workers=None):
    """Use ThreadPoolExecutor with optimal pool size"""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(func, *args) for _ in range(num_tasks)]
        for future in futures:
            future.result()

def run_process_pool(num_tasks: int, func, *args, max_workers=None):
    """Use ProcessPoolExecutor for CPU-bound tasks"""
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        task_data = [(func, args) for _ in range(num_tasks)]
        list(executor.map(worker_wrapper, task_data))

# =====================================================================
# BENCHMARK RUNNER
# =====================================================================
def run_benchmark(run_func, num_tasks: int, func, *args,
                 iterations: int = 3, **kwargs):
    """Run a single benchmark and return results"""
    process = psutil.Process(os.getpid())
    results = []

    for i in range(iterations):
        time.sleep(0.3)

        mem_before = process.memory_info().rss / (1024*1024)

        start = time.perf_counter()
        try:
            run_func(num_tasks, func, *args, **kwargs)
            elapsed = time.perf_counter() - start
        except Exception as e:
            print(f"  ❌ Iteration {i+1} failed: {e}")
            continue

        mem_after = process.memory_info().rss / (1024*1024)

        throughput = num_tasks / elapsed if elapsed > 0 else 0
        mem_delta = mem_after - mem_before

        results.append({
            'time': elapsed,
            'throughput': throughput,
            'memory': mem_delta,
        })

    if not results:
        return None

    times = [r['time'] for r in results]
    return {
        'avg_time': statistics.mean(times),
        'std_time': statistics.stdev(times) if len(times) > 1 else 0,
        'min_time': min(times),
        'max_time': max(times),
        'avg_throughput': statistics.mean([r['throughput'] for r in results]),
    }

# =====================================================================
# COMPREHENSIVE TEST SUITE
# =====================================================================
def run_all_tests(system_info: dict):
    """Run all benchmarks and return structured results"""
    cores = system_info['cpu']['cores_logical']
    results = {
        'system': system_info,
        'benchmarks': {}
    }

    print(f"\n{'='*70}")
    print(f"RUNNING BENCHMARKS ON: {system_info['system_name']}")
    print(f"CPU: {system_info['cpu']['brand']}")
    print(f"Cores: {cores} logical ({system_info['cpu']['cores_physical']} physical)")
    print(f"{'='*70}")

    # Prepare test data
    test_image = generate_test_image((800, 600))
    img_bytes = BytesIO()
    test_image.save(img_bytes, format='PNG')
    image_data = img_bytes.getvalue()

    # ----------------------------------------------------------------
    # TEST 1: IMAGE PROCESSING (CPU-BOUND, QUICK)
    # ----------------------------------------------------------------
    print(f"\n🎨 Test 1: Image Processing (30 images, 14 filters)")

    img_new = run_benchmark(run_new_threads, 30, image_processing_task, image_data, iterations=2)
    print(f"  ✓ New Threads: {img_new['avg_time']:.3f}s")

    img_thread = run_benchmark(run_thread_pool, 30, image_processing_task, image_data,
                               max_workers=cores, iterations=2)
    print(f"  ✓ ThreadPool: {img_thread['avg_time']:.3f}s")

    img_process = run_benchmark(run_process_pool, 30, image_processing_task, image_data,
                                max_workers=cores, iterations=2)
    print(f"  ✓ ProcessPool: {img_process['avg_time']:.3f}s")

    results['benchmarks']['image_processing'] = {
        'new_threads': img_new,
        'thread_pool': img_thread,
        'process_pool': img_process,
    }

    # ----------------------------------------------------------------
    # TEST 2: FACTORIAL (CPU-BOUND, HEAVY)
    # ----------------------------------------------------------------
    print(f"\n🔢 Test 2: Factorial Calculation (50 tasks, 50,000!)")

    fact_new = run_benchmark(run_new_threads, 50, factorial_task, 50000, iterations=2)
    print(f"  ✓ New Threads: {fact_new['avg_time']:.3f}s")

    fact_thread = run_benchmark(run_thread_pool, 50, factorial_task, 50000,
                                max_workers=cores, iterations=2)
    print(f"  ✓ ThreadPool: {fact_thread['avg_time']:.3f}s")

    fact_process = run_benchmark(run_process_pool, 50, factorial_task, 50000,
                                 max_workers=cores, iterations=2)
    print(f"  ✓ ProcessPool: {fact_process['avg_time']:.3f}s")

    results['benchmarks']['factorial'] = {
        'new_threads': fact_new,
        'thread_pool': fact_thread,
        'process_pool': fact_process,
    }

    # ----------------------------------------------------------------
    # TEST 3: WEB CRAWLING (IO-BOUND)
    # ----------------------------------------------------------------
    print(f"\n🌐 Test 3: Web Crawling (15 requests)")

    test_url = "https://www.ipsantarem.pt/"
    pool_size = cores * 2  # 2x cores for I/O

    web_new = run_benchmark(run_new_threads, 15, web_crawler_task, test_url, iterations=1)
    print(f"  ✓ New Threads: {web_new['avg_time']:.3f}s")
    time.sleep(0.5)

    web_thread = run_benchmark(run_thread_pool, 15, web_crawler_task, test_url,
                               max_workers=pool_size, iterations=1)
    print(f"  ✓ ThreadPool ({pool_size}): {web_thread['avg_time']:.3f}s")

    results['benchmarks']['web_crawling'] = {
        'new_threads': web_new,
        'thread_pool': web_thread,
    }

    # ----------------------------------------------------------------
    # TEST 4: SLEEP (IO-BOUND, LIGHT)
    # ----------------------------------------------------------------
    print(f"\n😴 Test 4: Sleep Tasks (100 tasks, 0.05s each)")

    pool_size = cores * 8  # 8x cores for light I/O

    sleep_new = run_benchmark(run_new_threads, 100, sleep_task, 0.05, iterations=2)
    print(f"  ✓ New Threads: {sleep_new['avg_time']:.3f}s")

    sleep_thread = run_benchmark(run_thread_pool, 100, sleep_task, 0.05,
                                 max_workers=pool_size, iterations=2)
    print(f"  ✓ ThreadPool ({pool_size}): {sleep_thread['avg_time']:.3f}s")

    results['benchmarks']['sleep'] = {
        'new_threads': sleep_new,
        'thread_pool': sleep_thread,
    }

    return results

# =====================================================================
# RESULTS SAVER
# =====================================================================
def save_results(results: dict):
    """Save results to JSON file with system name"""
    system_name = results['system']['system_name'].replace(' ', '_')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"benchmark_{system_name}_{timestamp}.json"

    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to: {filename}")
    return filename

# =====================================================================
# COMPARISON TOOL
# =====================================================================
def compare_systems(result_files: List[str]):
    """Compare results from multiple systems"""
    all_results = []

    # Load all result files
    for file in result_files:
        try:
            with open(file, 'r') as f:
                all_results.append(json.load(f))
        except Exception as e:
            print(f"Error loading {file}: {e}")
            continue

    if len(all_results) < 2:
        print("Need at least 2 result files to compare!")
        return

    print("\n" + "="*70)
    print("MULTI-SYSTEM COMPARISON")
    print("="*70)

    # System overview
    print("\n📊 SYSTEMS TESTED:")
    print("-"*70)
    for i, result in enumerate(all_results, 1):
        sys = result['system']
        print(f"{i}. {sys['system_name']}")
        print(f"   CPU: {sys['cpu']['brand']}")
        print(f"   Cores: {sys['cpu']['cores_logical']} logical ({sys['cpu']['cores_physical']} physical)")
        print(f"   RAM: {sys['memory']['total_gb']} GB")
        print(f"   OS: {sys['os']}")
        print()

    # Compare each benchmark
    benchmarks = ['image_processing', 'factorial', 'web_crawling', 'sleep']
    benchmark_names = {
        'image_processing': 'Image Processing (CPU-bound, quick)',
        'factorial': 'Factorial (CPU-bound, heavy)',
        'web_crawling': 'Web Crawling (I/O-bound)',
        'sleep': 'Sleep Tasks (I/O-bound, light)'
    }

    for bench in benchmarks:
        print("\n" + "="*70)
        print(f"📈 {benchmark_names[bench].upper()}")
        print("="*70)

        # Compare ProcessPool for CPU tasks, ThreadPool for I/O tasks
        if bench in ['image_processing', 'factorial']:
            approaches = ['new_threads', 'thread_pool', 'process_pool']
        else:
            approaches = ['new_threads', 'thread_pool']

        for approach in approaches:
            print(f"\n{approach.replace('_', ' ').title()}:")
            print("-"*70)

            times = []
            for result in all_results:
                if bench in result['benchmarks'] and approach in result['benchmarks'][bench]:
                    sys_name = result['system']['system_name']
                    time_val = result['benchmarks'][bench][approach]['avg_time']
                    cores = result['system']['cpu']['cores_logical']
                    times.append((sys_name, time_val, cores))

            # Sort by time
            times.sort(key=lambda x: x[1])

            if times:
                fastest = times[0]
                for sys_name, time_val, cores in times:
                    speedup = fastest[1] / time_val
                    marker = " 🏆 FASTEST" if time_val == fastest[1] else ""
                    print(f"  {sys_name:<30} {time_val:>8.3f}s ({cores} cores) [vs fastest: {1/speedup:.2f}x]{marker}")

    # Performance per core analysis
    print("\n" + "="*70)
    print("⚙️  PERFORMANCE PER CORE ANALYSIS")
    print("="*70)

    print("\nFactorial (ProcessPool) - Time per Core:")
    print("-"*70)
    for result in all_results:
        sys_name = result['system']['system_name']
        cores = result['system']['cpu']['cores_logical']
        time_val = result['benchmarks']['factorial']['process_pool']['avg_time']
        time_per_core = time_val * cores  # Total CPU-seconds

        print(f"  {sys_name:<30} {time_per_core:>8.2f} CPU-seconds")

    print("\n💡 Lower CPU-seconds = more efficient CPU")

# =====================================================================
# MAIN
# =====================================================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("MULTI-SYSTEM THREAD POOL BENCHMARK TOOL")
    print("="*70)

    print("\nThis tool will:")
    print("1. Collect system information")
    print("2. Run comprehensive benchmarks")
    print("3. Save results to JSON file")
    print("4. Allow comparison with other systems")

    choice = input("\n[1] Run benchmarks on this system\n[2] Compare existing results\nChoice: ")

    if choice == "1":
        # Collect system info
        try:
            system_info = SystemInfo.collect()
        except ImportError:
            print("\n⚠️  Installing py-cpuinfo for detailed CPU info...")
            print("Run: pip install py-cpuinfo")
            print("Using basic system info for now...")
            system_info = {
                "system_name": platform.node(),
                "os": f"{platform.system()} {platform.release()}",
                "python_version": platform.python_version(),
                "cpu": {
                    "brand": "Unknown (install py-cpuinfo for details)",
                    "cores_physical": psutil.cpu_count(logical=False),
                    "cores_logical": psutil.cpu_count(logical=True),
                    "frequency_mhz": 0,
                    "architecture": platform.machine(),
                },
                "memory": {
                    "total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                },
                "test_timestamp": datetime.now().isoformat(),
            }

        input("\nPress Enter to start benchmarks...")

        # Run tests
        results = run_all_tests(system_info)

        # Save results
        filename = save_results(results)

        print("\n" + "="*70)
        print("✅ BENCHMARKS COMPLETE!")
        print("="*70)
        print(f"\nResults file: {filename}")
        print("\nRun this script on other laptops and use option [2]")
        print("to compare results!")

    elif choice == "2":
        print("\nEnter JSON result files to compare (comma-separated):")
        print("Example: laptop1.json, laptop2.json, laptop3.json")
        files_input = input("\nFiles: ")

        files = [f.strip() for f in files_input.split(',')]
        compare_systems(files)

    else:
        print("Invalid choice!")