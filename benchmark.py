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
        try:
            import cpuinfo
            cpu_info = cpuinfo.get_cpu_info()
            cpu_brand = cpu_info.get('brand_raw', 'Unknown')
        except ImportError:
            cpu_brand = "Unknown (install py-cpuinfo for details)"

        # Memory information
        memory = psutil.virtual_memory()

        # Python version
        python_version = platform.python_version()

        return {
            "system_name": platform.node(),
            "os": f"{platform.system()} {platform.release()}",
            "python_version": python_version,
            "cpu": {
                "brand": cpu_brand,
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
def generate_test_image(size=(3840, 2160)):  # 4K resolution
    """Generate a colorful test image for processing"""
    img = Image.new('RGB', size, color=(135, 206, 235))
    draw = ImageDraw.Draw(img)

    # Scale factors for 4K
    scale_x = size[0] / 1920
    scale_y = size[1] / 1080

    # Sun
    draw.ellipse([int(1600*scale_x), int(100*scale_y), int(1850*scale_x), int(350*scale_y)],
                 fill=(255, 255, 0))
    # Ground
    draw.rectangle([0, int(800*scale_y), size[0], size[1]], fill=(34, 139, 34))
    # House
    draw.rectangle([int(400*scale_x), int(600*scale_y), int(800*scale_x), int(1000*scale_y)],
                   fill=(139, 69, 19))
    draw.polygon([(int(400*scale_x), int(600*scale_y)),
                  (int(600*scale_x), int(400*scale_y)),
                  (int(800*scale_x), int(600*scale_y))], fill=(178, 34, 34))
    # Windows
    draw.rectangle([int(460*scale_x), int(700*scale_y), int(540*scale_x), int(800*scale_y)],
                   fill=(173, 216, 230))
    draw.rectangle([int(660*scale_x), int(700*scale_y), int(740*scale_x), int(800*scale_y)],
                   fill=(173, 216, 230))
    # Door
    draw.rectangle([int(560*scale_x), int(840*scale_y), int(640*scale_x), int(1000*scale_y)],
                   fill=(101, 67, 33))
    # Trees
    draw.rectangle([int(1000*scale_x), int(760*scale_y), int(1040*scale_x), int(1000*scale_y)],
                   fill=(101, 67, 33))
    draw.ellipse([int(920*scale_x), int(640*scale_y), int(1120*scale_x), int(840*scale_y)],
                 fill=(0, 100, 0))
    draw.rectangle([int(200*scale_x), int(760*scale_y), int(240*scale_x), int(1000*scale_y)],
                   fill=(101, 67, 33))
    draw.ellipse([int(120*scale_x), int(640*scale_y), int(320*scale_x), int(840*scale_y)],
                 fill=(0, 128, 0))

    return img

def image_processing_task(image_data: bytes) -> bytes:
    """EXTREMELY HEAVY CPU-intensive image processing - 50 filters on 4K image"""
    img = Image.open(BytesIO(image_data))

    # Round 1: Heavy blurs (10 filters)
    for radius in [15, 12, 10, 8, 15, 12, 10, 8, 6, 5]:
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))

    # Round 2: Enhancements (10 filters)
    img = ImageEnhance.Contrast(img).enhance(2.0)
    img = ImageEnhance.Brightness(img).enhance(1.3)
    img = ImageEnhance.Color(img).enhance(1.8)
    img = ImageEnhance.Sharpness(img).enhance(2.5)
    img = ImageEnhance.Contrast(img).enhance(1.5)
    img = ImageEnhance.Brightness(img).enhance(1.1)
    img = ImageEnhance.Color(img).enhance(1.3)
    img = ImageEnhance.Sharpness(img).enhance(1.8)
    img = ImageEnhance.Contrast(img).enhance(1.2)
    img = ImageEnhance.Brightness(img).enhance(1.05)

    # Round 3: Edge detection and smoothing (10 filters)
    img = img.filter(ImageFilter.FIND_EDGES)
    img = img.filter(ImageFilter.SMOOTH_MORE)
    img = img.filter(ImageFilter.SMOOTH)
    img = img.filter(ImageFilter.SMOOTH_MORE)
    img = img.filter(ImageFilter.SMOOTH)
    img = img.filter(ImageFilter.FIND_EDGES)
    img = img.filter(ImageFilter.SMOOTH_MORE)
    img = img.filter(ImageFilter.SMOOTH)
    img = img.filter(ImageFilter.SMOOTH_MORE)
    img = img.filter(ImageFilter.SMOOTH)

    # Round 4: Detail and edge enhancement (10 filters)
    img = img.filter(ImageFilter.DETAIL)
    img = img.filter(ImageFilter.EDGE_ENHANCE)
    img = img.filter(ImageFilter.EDGE_ENHANCE_MORE)
    img = img.filter(ImageFilter.DETAIL)
    img = img.filter(ImageFilter.EDGE_ENHANCE)
    img = img.filter(ImageFilter.EDGE_ENHANCE_MORE)
    img = img.filter(ImageFilter.DETAIL)
    img = img.filter(ImageFilter.EDGE_ENHANCE)
    img = img.filter(ImageFilter.EDGE_ENHANCE_MORE)
    img = img.filter(ImageFilter.DETAIL)

    # Round 5: Final heavy blurs and enhancements (10 filters)
    img = img.filter(ImageFilter.GaussianBlur(radius=10))
    img = img.filter(ImageFilter.GaussianBlur(radius=8))
    img = img.filter(ImageFilter.GaussianBlur(radius=6))
    img = ImageEnhance.Sharpness(img).enhance(2.0)
    img = img.filter(ImageFilter.GaussianBlur(radius=5))
    img = img.filter(ImageFilter.GaussianBlur(radius=4))
    img = ImageEnhance.Contrast(img).enhance(1.3)
    img = img.filter(ImageFilter.GaussianBlur(radius=3))
    img = ImageEnhance.Color(img).enhance(1.2)
    img = img.filter(ImageFilter.SHARPEN)

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
# BENCHMARK RUNNER WITH ALL 4 CRITERIA
# =====================================================================
def run_benchmark(run_func, num_tasks: int, func, *args,
                 iterations: int = 3, **kwargs):
    """Run benchmark and measure ALL 4 CRITERIA:
    1. Execution time
    2. Memory used
    3. CPU usage
    4. Tasks completed per second (throughput)
    """
    process = psutil.Process(os.getpid())
    results = []

    for i in range(iterations):
        time.sleep(0.3)  # Cooldown

        # CRITERION 2: Memory BEFORE
        mem_before = process.memory_info().rss / (1024*1024)  # MB

        # CRITERION 3: CPU usage monitoring (sample at start)
        cpu_percent_before = psutil.cpu_percent(interval=None)

        # CRITERION 1: Execution time START
        start = time.perf_counter()

        # Start CPU monitoring thread
        cpu_samples = []
        stop_monitoring = threading.Event()

        def monitor_cpu():
            while not stop_monitoring.is_set():
                cpu_samples.append(psutil.cpu_percent(interval=0.1))

        monitor_thread = threading.Thread(target=monitor_cpu, daemon=True)
        monitor_thread.start()

        try:
            run_func(num_tasks, func, *args, **kwargs)
            elapsed = time.perf_counter() - start  # CRITERION 1: Execution time
        except Exception as e:
            print(f"  X Iteration {i+1} failed: {e}")
            stop_monitoring.set()
            continue
        finally:
            stop_monitoring.set()
            monitor_thread.join(timeout=1)

        # CRITERION 2: Memory AFTER
        mem_after = process.memory_info().rss / (1024*1024)  # MB
        mem_delta = mem_after - mem_before

        # CRITERION 3: Average CPU usage during execution
        avg_cpu = statistics.mean(cpu_samples) if cpu_samples else 0.0

        # CRITERION 4: Tasks per second (throughput)
        throughput = num_tasks / elapsed if elapsed > 0 else 0

        results.append({
            'time': elapsed,              # CRITERION 1
            'memory_mb': mem_delta,       # CRITERION 2
            'cpu_percent': avg_cpu,       # CRITERION 3
            'throughput': throughput,     # CRITERION 4
        })

        print(f"    Iteration {i+1}: {elapsed:.3f}s | {mem_delta:.1f}MB | {avg_cpu:.1f}% CPU | {throughput:.2f} tasks/s")

    if not results:
        return None

    times = [r['time'] for r in results]
    mems = [r['memory_mb'] for r in results]
    cpus = [r['cpu_percent'] for r in results]
    throughputs = [r['throughput'] for r in results]

    return {
        # CRITERION 1: Execution time
        'avg_time': statistics.mean(times),
        'std_time': statistics.stdev(times) if len(times) > 1 else 0,
        'min_time': min(times),
        'max_time': max(times),

        # CRITERION 2: Memory used
        'avg_memory_mb': statistics.mean(mems),
        'max_memory_mb': max(mems),

        # CRITERION 3: CPU usage
        'avg_cpu_percent': statistics.mean(cpus),
        'max_cpu_percent': max(cpus),

        # CRITERION 4: Tasks per second
        'avg_throughput': statistics.mean(throughputs),
        'max_throughput': max(throughputs),
    }

# =====================================================================
# COMPREHENSIVE TEST SUITE
# =====================================================================
def run_all_tests(system_info: dict):
    """Run all benchmarks and return structured results"""
    cores = system_info['cpu']['cores_logical']
    results = {
        'system': system_info,
        'benchmarks': {},
        'criteria_measured': [
            'Execution Time (seconds)',
            'Memory Used (MB)',
            'CPU Usage (%)',
            'Tasks Completed per Second (throughput)'
        ]
    }

    print(f"\n{'='*70}")
    print(f"RUNNING BENCHMARKS ON: {system_info['system_name']}")
    print(f"CPU: {system_info['cpu']['brand']}")
    print(f"Cores: {cores} logical ({system_info['cpu']['cores_physical']} physical)")
    print(f"{'='*70}")

    print(f"\nMEASURING 4 CRITERIA:")
    print(f"  1. Execution Time (seconds)")
    print(f"  2. Memory Used (MB)")
    print(f"  3. CPU Usage (%)")
    print(f"  4. Tasks Completed per Second (throughput)")

    # Prepare test data
    test_image = generate_test_image((3840, 2160))  # 4K image
    img_bytes = BytesIO()
    test_image.save(img_bytes, format='PNG')
    image_data = img_bytes.getvalue()

    # Save original and processed images
    print(f"\nPreparing test images (3840x2160 pixels 4K, {len(image_data)/1024:.1f} KB)")
    test_image.save("original_test_image_4k.png")
    print(f"   OK Original image saved: original_test_image_4k.png")

    print(f"   Processing sample image to show results...")
    processed_bytes = image_processing_task(image_data)
    processed_img = Image.open(BytesIO(processed_bytes))
    processed_img.save("processed_test_image_4k.png")
    print(f"   OK Processed image saved: processed_test_image_4k.png")

    # ----------------------------------------------------------------
    # TEST 1: IMAGE PROCESSING (CPU-BOUND, EXTREMELY HEAVY)
    # ----------------------------------------------------------------
    print(f"\nTest 1: Image Processing (100 images, 50 filters, 4K resolution)")

    print(f"\n  New Threads:")
    img_new = run_benchmark(run_new_threads, 100, image_processing_task, image_data, iterations=2)

    print(f"\n  ThreadPool ({cores} workers):")
    img_thread = run_benchmark(run_thread_pool, 100, image_processing_task, image_data,
                               max_workers=cores, iterations=2)

    print(f"\n  ProcessPool ({cores} workers):")
    img_process = run_benchmark(run_process_pool, 100, image_processing_task, image_data,
                                max_workers=cores, iterations=2)

    results['benchmarks']['image_processing_4k'] = {
        'new_threads': img_new,
        'thread_pool': img_thread,
        'process_pool': img_process,
    }

    # ----------------------------------------------------------------
    # TEST 2: FACTORIAL (CPU-BOUND, HEAVY)
    # ----------------------------------------------------------------
    print(f"\nTest 2: Factorial Calculation (100 tasks, 100,000!)")

    print(f"\n  New Threads:")
    fact_new = run_benchmark(run_new_threads, 100, factorial_task, 100000, iterations=2)

    print(f"\n  ThreadPool ({cores} workers):")
    fact_thread = run_benchmark(run_thread_pool, 100, factorial_task, 100000,
                                max_workers=cores, iterations=2)

    print(f"\n  ProcessPool ({cores} workers):")
    fact_process = run_benchmark(run_process_pool, 100, factorial_task, 100000,
                                 max_workers=cores, iterations=2)

    results['benchmarks']['factorial'] = {
        'new_threads': fact_new,
        'thread_pool': fact_thread,
        'process_pool': fact_process,
    }

    # ----------------------------------------------------------------
    # TEST 3: WEB CRAWLING (IO-BOUND)
    # ----------------------------------------------------------------
    print(f"\nTest 3: Web Crawling (20 requests)")

    test_url = "https://www.ipsantarem.pt/"
    pool_size = cores * 4

    print(f"\n  New Threads:")
    web_new = run_benchmark(run_new_threads, 20, web_crawler_task, test_url, iterations=1)

    time.sleep(0.5)

    print(f"\n  ThreadPool ({pool_size} workers):")
    web_thread = run_benchmark(run_thread_pool, 20, web_crawler_task, test_url,
                               max_workers=pool_size, iterations=1)

    results['benchmarks']['web_crawling'] = {
        'new_threads': web_new,
        'thread_pool': web_thread,
    }

    # ----------------------------------------------------------------
    # TEST 4: SLEEP (IO-BOUND, LIGHT)
    # ----------------------------------------------------------------
    print(f"\nTest 4: Sleep Tasks (200 tasks, 0.05s each)")

    pool_size = cores * 4

    print(f"\n  New Threads:")
    sleep_new = run_benchmark(run_new_threads, 200, sleep_task, 0.05, iterations=2)

    print(f"\n  ThreadPool ({pool_size} workers):")
    sleep_thread = run_benchmark(run_thread_pool, 200, sleep_task, 0.05,
                                 max_workers=pool_size, iterations=2)

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

    print(f"\nResults saved to: {filename}")
    return filename

# =====================================================================
# MAIN
# =====================================================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("MULTI-SYSTEM BENCHMARK TOOL")
    print("Measuring: Time | Memory | CPU | Throughput")
    print("="*70)

    # Collect system info
    system_info = SystemInfo.collect()

    # Run tests
    results = run_all_tests(system_info)

    # Save results
    filename = save_results(results)

    print("\n" + "="*70)
    print("BENCHMARKS COMPLETE!")
    print("="*70)
    print(f"Results saved to: {filename}")