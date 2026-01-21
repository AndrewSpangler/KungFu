import numpy as np
import math
import time
import textwrap
from panda3d.core import (
    NodePath, Shader, ShaderAttrib, ShaderBuffer, 
    GeomEnums, ComputeNode, load_prc_file_data,
    GraphicsPipeSelection, FrameBufferProperties,
    WindowProperties, GraphicsPipe
)
from direct.showbase.ShowBase import ShowBase
from ..cast_buffer import CastBuffer

class Radix16FFT:
    def __init__(self, showbase, headless=False):
        self.showbase = showbase
        self._setup_context(headless)
        
        # Compile FFT Shaders
        self.digit_reverse_node = self._compile(self.digit_reverse_shader)
        self.butterfly_node = self._compile(self.butterfly_shader)

    def _setup_context(self, headless=False):
        if headless:
            graphics_pipe = GraphicsPipeSelection.get_global_ptr().make_default_pipe()
            framebuffer_properties = FrameBufferProperties()
            window_properties = WindowProperties.size(1, 1)
            self.showbase.win = self.showbase.graphics_engine.make_output(
                graphics_pipe, "fft16_headless", 0, framebuffer_properties, 
                window_properties, GraphicsPipe.BF_refuse_window
            )

    def _compile(self, shader_code):
        shader = Shader.make_compute(Shader.SL_GLSL, shader_code)
        node = NodePath(ComputeNode("fft16_op"))
        node.set_shader(shader)
        return node

    def push(self, data):
        data = np.ascontiguousarray(data, dtype=np.complex64)
        shader_buffer = ShaderBuffer("Data", data.tobytes(), GeomEnums.UH_stream)
        return CastBuffer(shader_buffer, len(data), cast=np.complex64)

    def fetch(self, gpu_handle):
        graphics_state = self.showbase.win.get_gsg()
        raw_data = self.showbase.graphics_engine.extract_shader_buffer_data(
            gpu_handle.buffer, graphics_state
        )
        return np.frombuffer(raw_data, dtype=gpu_handle.cast)

    def fft(self, data, inverse=False):
        buffer_handle = data if isinstance(data, CastBuffer) else self.push(data)
        num_elements = buffer_handle.n_items

        num_stages = int(math.log(num_elements) / math.log(16))
        if 16 ** num_stages != num_elements:
            raise ValueError(f"Size must be power of 16, got {num_elements}")
        
        inverse_flag = -1 if inverse else 1

        digit_reversed_buffer = ShaderBuffer("DR_Out", num_elements * 8, GeomEnums.UH_stream)
        self.digit_reverse_node.set_shader_input("DataIn", buffer_handle.buffer)
        self.digit_reverse_node.set_shader_input("DataOut", digit_reversed_buffer)
        self.digit_reverse_node.set_shader_input("nItems", int(num_elements))
        self.digit_reverse_node.set_shader_input("log16N", int(num_stages))
        
        self.showbase.graphics_engine.dispatch_compute(
            ((num_elements + 63) // 64, 1, 1), 
            self.digit_reverse_node.get_attrib(ShaderAttrib), 
            self.showbase.win.get_gsg()
        )
        
        current_input_buffer = digit_reversed_buffer

        self.butterfly_node.set_shader_input("nItems", int(num_elements))
        self.butterfly_node.set_shader_input("inverse", inverse_flag)

        for stage_index in range(num_stages):
            stage_size = 16 ** stage_index
            stage_output_buffer = ShaderBuffer(f"Stage_{stage_index}", num_elements * 8, GeomEnums.UH_stream)
            
            self.butterfly_node.set_shader_input("DataIn", current_input_buffer)
            self.butterfly_node.set_shader_input("DataOut", stage_output_buffer)
            self.butterfly_node.set_shader_input("stage", int(stage_size))
            num_work_items = (num_elements // 16) 
            self.showbase.graphics_engine.dispatch_compute(
                ((num_work_items + 63) // 64, 1, 1), 
                self.butterfly_node.get_attrib(ShaderAttrib), 
                self.showbase.win.get_gsg()
            )
            current_input_buffer = stage_output_buffer

        result_handle = CastBuffer(current_input_buffer, num_elements, cast=np.complex64)
        
        if inverse:
            return self.fetch(result_handle) / num_elements
        return result_handle

    @property
    def digit_reverse_shader(self) -> str:
        return textwrap.dedent("""
            #version 430
            layout (local_size_x = 64) in;
            layout(std430) buffer DataIn { vec2 data_in[]; };
            layout(std430) buffer DataOut { vec2 data_out[]; };

            uniform uint nItems;
            uniform uint log16N;

            uint reverse_digits_base16(uint value, uint num_digits) {
                uint reversed = 0;
                for (uint digit_index = 0; digit_index < num_digits; digit_index++) {
                    reversed = (reversed << 4) | (value & 0xF);
                    value >>= 4;
                }
                return reversed;
            }

            void main() {
                uint global_id = gl_GlobalInvocationID.x;
                if (global_id >= nItems) return;
                uint target_index = reverse_digits_base16(global_id, log16N);
                data_out[target_index] = data_in[global_id];
            }""")

    @property
    def butterfly_shader(self) -> str:
        return textwrap.dedent("""
        #version 430
        layout (local_size_x = 64) in;
        layout(std430) buffer DataIn { vec2 data_in[]; };
        layout(std430) buffer DataOut { vec2 data_out[]; };

        uniform uint nItems;
        uniform uint stage;
        uniform int inverse;

        const float PI = 3.14159265358979323846;

        vec2 complex_mul(vec2 a, vec2 b) {
            return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
        }

        void main() {
            uint global_id = gl_GlobalInvocationID.x;
            uint elements_per_group = stage * 16;
            uint num_groups = nItems / elements_per_group;
            
            if (global_id >= num_groups * stage) return;
            
            uint group_id = global_id / stage;
            uint element_in_stage = global_id % stage;
            uint base_index = group_id * elements_per_group + element_in_stage;
            
            vec2 temp_values[16];
            for (uint radix_index = 0; radix_index < 16; radix_index++) {
                temp_values[radix_index] = data_in[base_index + radix_index * stage];
            }
            
            vec2 result_values[16];
            for (uint output_index = 0; output_index < 16; output_index++) {
                result_values[output_index] = vec2(0.0, 0.0);
                for (uint input_index = 0; input_index < 16; input_index++) {
                    float radix_angle = -1.0 * inverse * 2.0 * PI * float(output_index * input_index) / 16.0;
                    float stage_angle = -1.0 * inverse * 2.0 * PI * float(element_in_stage * input_index) / float(elements_per_group);
                    float total_angle = radix_angle + stage_angle;
                    vec2 twiddle_factor = vec2(cos(total_angle), sin(total_angle));
                    result_values[output_index] += complex_mul(temp_values[input_index], twiddle_factor);
                }
            }
            
            for (uint output_index = 0; output_index < 16; output_index++) {
                data_out[base_index + output_index * stage] = result_values[output_index];
            }
        }""")

class FFT16Demo(ShowBase):
    def __init__(self, headless=True):
        load_prc_file_data("", "window-type none\naudio-library-name null")
        ShowBase.__init__(self)
        self.fft_engine = Radix16FFT(self, headless=headless)

    def run_test(self, num_points=65536, output=True):
        def out(*args, **kw):
            if output:
                print(*args, **kw)

        out(f"Testing Radix-16 {num_points}-point FFT...")
        
        # Generate test signal
        time_samples = np.linspace(0, 1, num_points)
        signal = np.sin(2 * np.pi * 50 * time_samples) + 0.5 * np.sin(2 * np.pi * 120 * time_samples)
        signal = signal + 0.2 * np.random.randn(num_points)
        signal = signal.astype(np.complex64)

        # GPU
        gpu_start_time = time.perf_counter()
        gpu_result_handle = self.fft_engine.fft(signal)
        gpu_result = self.fft_engine.fetch(gpu_result_handle)
        gpu_elapsed_time = time.perf_counter() - gpu_start_time

        # CPU (reference)
        cpu_start_time = time.perf_counter()
        cpu_result = np.fft.fft(signal)
        cpu_elapsed_time = time.perf_counter() - cpu_start_time

        # Results
        out("\n" + "="*90)
        out(f"{'Index':<8} | {'GPU Value':<30} | {'CPU Value':<30} | {'Abs Diff':<12}")
        out("-" * 90)
        for sample_index in range(100, 120): 
            gpu_value = gpu_result[sample_index]
            cpu_value = cpu_result[sample_index]
            difference = abs(gpu_value - cpu_value)
            out(f"{sample_index:<8} | {str(gpu_value):<30} | {str(cpu_value):<30} | {difference:.3e}")
        out("="*90 + "\n")

        max_difference = np.max(np.abs(gpu_result - cpu_result))
        mean_difference = np.mean(np.abs(gpu_result - cpu_result))

        out(f"GPU Time:  {gpu_elapsed_time:.5f}s")
        out(f"CPU Time:  {cpu_elapsed_time:.5f}s")
        out(f"Speedup:   {cpu_elapsed_time/gpu_elapsed_time:.2f}x")
        out(f"Max Diff:  {max_difference:.3e}")
        out(f"Mean Diff: {mean_difference:.3e}")
        out(f"Valid:     {max_difference < 1e-1}")
    
        out("\nTesting Inverse FFT...")
        # Inverse FFT
        inverse_start_time = time.perf_counter()
        inverse_result = self.fft_engine.fft(gpu_result_handle, inverse=True)
        inverse_elapsed_time = time.perf_counter() - inverse_start_time
        
        inverse_difference = np.max(np.abs(inverse_result - signal))
        out(f"IFFT Time:     {inverse_elapsed_time:.5f}s")
        out(f"Roundtrip Diff: {inverse_difference:.3e}")
        out(f"IFFT Valid:    {inverse_difference < 1e-1}")

        return gpu_elapsed_time, cpu_elapsed_time

if __name__ == "__main__":
    # Test with different sizes
    # sizes = [16**2, 16**3, 16**4, 16**5]  # 256, 4096, 65536
    
    # for size in sizes:
    #     print(f"\n{'='*90}")
    #     print(f"Testing N = {size}")
    #     print('='*90)
    #     demo_app = FFT16Demo(headless=True)
    #     demo_app.run_test(num_points=size)
    #     demo_app.destroy()
    #     print()

    #     sizes = [16**2, 16**3, 16**4, 16**5]  # 256, 4096, 65536
    
    total_gpu_time = 0
    total_cpu_time = 0
    num_iterations = 10
    for iteration_index in range(num_iterations):
        print(f"\n{'='*90}")
        print(f"Testing N = {16**5}")
        print('='*90)
        demo_app = FFT16Demo(headless=True)
        gpu_time, cpu_time = demo_app.run_test(num_points=16**5, output=False)
        total_gpu_time += gpu_time
        total_cpu_time += cpu_time
        demo_app.destroy()
        print()
    print(f"AVG GPU Time:  {total_gpu_time / num_iterations:.5f}s")
    print(f"AVG CPU Time:  {total_cpu_time / num_iterations:.5f}s")
    print(f"Speedup:   {total_cpu_time/total_gpu_time:.2f}x")