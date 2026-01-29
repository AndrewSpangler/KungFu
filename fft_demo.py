import time
import numpy as np
from kungfu import GPUMath, Radix16FFT
from direct.showbase.ShowBase import ShowBase
from panda3d.core import load_prc_file_data

class FFT16Demo(ShowBase):
    def __init__(self, headless=True):
        ShowBase.__init__(self)
        self.engine = GPUMath(self, headless=True)
        self.fft_class = Radix16FFT(self.engine)

    def run_test(self, N=65536, output=True, test_inverse=True):
        def out(*args, **kw):
            if output:
                print(*args, **kw)

        out(f"Testing Radix-16 {N}-point FFT...")
        
        # Generate test signal
        t = np.linspace(0, 1, N, endpoint=False)
        x = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)
        x = x + 0.2 * np.random.randn(N)
        x = x.astype(np.complex64)

        # GPU
        t0 = time.perf_counter()
        g_res_handle = self.fft_class.fft(x)
        final_gpu = self.engine.fetch(g_res_handle)
        t_gpu = time.perf_counter() - t0

        # CPU (reference)
        t1 = time.perf_counter()
        final_cpu = np.fft.fft(x)
        t_cpu = time.perf_counter() - t1

        # Results
        out("\n" + "="*90)
        out(f"{'Index':<8} | {'GPU Value':<30} | {'CPU Value':<30} | {'Abs Diff':<12}")
        out("-" * 90)
        for i in range(min(20, N)):
            g = final_gpu[i]
            c = final_cpu[i]
            diff = abs(g - c)
            out(f"{i:<8} | {str(g):<30} | {str(c):<30} | {diff:.3e}")
        out("="*90 + "\n")

        max_diff = np.max(np.abs(final_gpu - final_cpu))
        mean_diff = np.mean(np.abs(final_gpu - final_cpu))

        out(f"GPU Time:  {t_gpu:.5f}s")
        out(f"CPU Time:  {t_cpu:.5f}s")
        out(f"Speedup:   {t_cpu/t_gpu:.2f}x")
        out(f"Max Diff:  {max_diff:.3e}")
        out(f"Mean Diff: {mean_diff:.3e}")
        out(f"Valid:     {max_diff < 1e-1}")
    
        # Inverse FFT
        if test_inverse:
            out("\nTesting Inverse FFT...")
            t2 = time.perf_counter()
            final_inv_handle = self.fft_class.fft(g_res_handle, inverse=True)
            final_inv = self.engine.fetch(final_inv_handle)
            t_inv = time.perf_counter() - t2
            inv_diff = np.max(np.abs(final_inv - x))
            out(f"IFFT Time:     {t_inv:.5f}s")
            out(f"Roundtrip Diff: {inv_diff:.3e}")
            out(f"IFFT Valid:    {inv_diff < 1e-1}")

        return t_gpu, t_cpu

def benchmark_gpu_avg(count = 1):
    for k, v in {
        "window-type":          "none",
        "audio-library-name":   "null",
        "sync-video":           "#f",
    }.items():
        load_prc_file_data("", f"{k} {v}")

    app = FFT16Demo(headless=True)
    gpu_times = []
    cpu_times = []
    
    # Test with smaller size first
    print("Testing with smaller size first...")
    app.run_test(N=256, output=True, test_inverse=True)
    
    # Then benchmark with larger size
    print("\n" + "="*60)
    print("Starting benchmark...")
    print("="*60)
    
    for i in range(count + 1):
        t_gpu, t_cpu = app.run_test(N=16**5, output=True, test_inverse=False)
        gpu_times.append(t_gpu)
        cpu_times.append(t_cpu)
        print(f"Run {i+1}: GPU={t_gpu:.5f}s, CPU={t_cpu:.5f}s, Speedup={t_cpu/t_gpu:.2f}x")
        app.task_mgr.step()

    # Discard warm-up 
    gpu_times = gpu_times[1:]
    cpu_times = cpu_times[1:]

    app.destroy()
    total_gpu   = sum(gpu_times)
    avg_gpu     = total_gpu / len(gpu_times)
    max_gpu     = max(gpu_times)
    min_gpu     = min(gpu_times)
    
    total_cpu   = sum(cpu_times)
    avg_cpu     = total_cpu / len(cpu_times)
    max_cpu     = max(cpu_times)
    min_cpu     = min(cpu_times)
    
    print(f"\n{'='*60}")
    print(f"BENCHMARK RESULTS ({len(gpu_times)} runs after warm-up):")
    print(f"{'='*60}")
    print(f"GPU: AVG={avg_gpu:.5f}s | MIN={min_gpu:.5f}s | MAX={max_gpu:.5f}s")
    print(f"CPU: AVG={avg_cpu:.5f}s | MIN={min_cpu:.5f}s | MAX={max_cpu:.5f}s")
    print(f"Speedup: {avg_cpu/avg_gpu:.2f}x")

if __name__ == "__main__":
    benchmark_gpu_avg(count=1)