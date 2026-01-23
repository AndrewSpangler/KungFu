import numpy as np
import traceback
from direct.showbase.ShowBase import ShowBase
from gpu_math import GPUMath, gpu_kernel, inline_always, NP_GLTypes, IOTypes
from gpu_math.error_handler import print_compilation_error, CompilationError

class ErrorHandlingDemo(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)
        self.math = GPUMath(self, headless=True)
    
    def run_test(self, name, test_func, should_pass=True):
        """Run a test and handle errors appropriately"""
        print(f"\n{'='*60}")
        print(f"Test: {name}")
        print(f"Expected: {'PASS' if should_pass else 'FAIL with error'}")
        print(f"{'='*60}")
        
        try:
            result = test_func()
            if should_pass:
                print(f"✓ Test PASSED as expected")
                return True
            else:
                print(f"✗ Test FAILED: Expected error but passed")
                return False
        except CompilationError as e:
            print_compilation_error(e, show_traceback=True)
            if should_pass:
                print(f"✗ Test FAILED: Unexpected error")
                return False
            else:
                print(f"✓ Test PASSED: Error caught as expected")
                return True
        except Exception as e:
            print(f"✗ Unexpected exception type: {type(e).__name__}")
            print(f"Error: {e}")
            traceback.print_exc()
            return False
    
    # -----------------------------------------------------------------
    # POSITIVE TESTS (Should Pass)
    # -----------------------------------------------------------------
    
    def test_basic_kernel(self):
        """Basic kernel that should compile successfully"""
        @gpu_kernel
        def simple_add(a, b):
            return a + b
        
        compiled = self.math.compile_fused(simple_add, debug=True)
        x = np.array([1, 2, 3], dtype=np.float32)
        y = np.array([4, 5, 6], dtype=np.float32)
        result = compiled(x, y)
        output = self.math.fetch(result)
        print(f"Result: {output}")
        return True
    
    def test_typed_variables(self):
        """Kernel with typed variables"""
        @gpu_kernel
        def typed_kernel(a, b):
            x: float = a + 1.0
            y: int = int(b)
            return x * float(y)
        
        compiled = self.math.compile_fused(typed_kernel, debug=True)
        return True
    
    def test_inline_function(self):
        """Kernel with inline function"""
        @gpu_kernel
        def inline_test(a):
            @inline_always
            def square(x):
                return x * x
            
            return square(a) + 1.0
        
        compiled = self.math.compile_fused(inline_test, debug=True)
        return True
    
    def test_for_loop(self):
        """Kernel with for loop"""
        @gpu_kernel
        def loop_kernel(a):
            result: float = 0.0
            for i in range(10):
                result = result + a * float(i)
            return result
        
        compiled = self.math.compile_fused(loop_kernel, debug=True)
        return True
    
    def test_if_statement(self):
        """Kernel with if statement"""
        @gpu_kernel
        def conditional_kernel(a, b):
            if a > b:
                return a * 2.0
            else:
                return b * 3.0
        
        compiled = self.math.compile_fused(conditional_kernel, debug=True)
        return True
    
    def test_nested_loops(self):
        """Kernel with nested loops"""
        @gpu_kernel
        def nested_loops(a):
            total: float = 0.0
            for i in range(3):
                for j in range(4):
                    total = total + a * float(i + j)
            return total
        
        compiled = self.math.compile_fused(nested_loops, debug=True)
        return True
    
    def test_builtin_functions(self):
        """Kernel using built-in math functions"""
        @gpu_kernel
        def math_functions(a):
            return sin(a) * cos(a) + sqrt(abs(a))
        
        compiled = self.math.compile_fused(math_functions, debug=True)
        return True
    
    # -----------------------------------------------------------------
    # NEGATIVE TESTS (Should Fail with Specific Errors)
    # -----------------------------------------------------------------
    
    def test_undefined_variable(self):
        """Using undefined variable should fail"""
        @gpu_kernel
        def undefined_var(a):
            return a + undefined_variable  # This variable doesn't exist
        
        compiled = self.math.compile_fused(undefined_var, debug=True)
            
    def test_invalid_for_loop(self):
        """For loop with non-range iterator"""
        @gpu_kernel
        def invalid_loop(a):
            for i in [1, 2, 3]:  # List iteration not supported
                a = a + i
            return a
        
        compiled = self.math.compile_fused(invalid_loop, debug=True)
        return False
    
    def test_missing_return(self):
        """Function without return in code path"""
        @gpu_kernel
        def missing_return(a):
            if a > 0:
                return a * 2.0
            # Missing return for a <= 0
        
        compiled = self.math.compile_fused(missing_return, debug=True)
        return False
    
    def test_unsupported_syntax(self):
        """Using unsupported Python syntax"""
        @gpu_kernel
        def unsupported_syntax(a):
            # Try/except not supported
            try:
                return a * 2
            except:
                return 0.0
        
        compiled = self.math.compile_fused(unsupported_syntax, debug=True)
        return False
    
    def test_invalid_function_call(self):
        """Calling undefined function"""
        @gpu_kernel
        def invalid_call(a):
            return my_undefined_function(a)  # This function doesn't exist
        
        compiled = self.math.compile_fused(invalid_call, debug=True)
        return False
    
    def test_complex_assignment(self):
        """Complex assignment not supported"""
        @gpu_kernel
        def complex_assignment(a, b):
            x, y = a, b  # Tuple assignment not supported
            return x + y
        
        compiled = self.math.compile_fused(complex_assignment, debug=True)
        return False
    
    def test_while_loop_error(self):
        """While loop with potential infinite loop"""
        @gpu_kernel
        def while_loop_error(a):
            i: int = 0
            while i < a:  # 'a' is array, not scalar - semantic error
                i = i + 1
            return float(i)
        
        compiled = self.math.compile_fused(while_loop_error, debug=True)
        return False
        
    def test_nested_function_error(self):
        """Nested function with errors"""
        @gpu_kernel
        def nested_error(a):
            def inner_func(x):
                return x + undefined  # Error inside nested function
            
            return inner_func(a)
        
        compiled = self.math.compile_fused(nested_error, debug=True)
        return False
    
    def test_array_indexing(self):
        """Array indexing not supported"""
        @gpu_kernel
        def array_index(a):
            arr = [1.0, 2.0, 3.0]  # List literal not supported
            return a + arr[0]
        
        compiled = self.math.compile_fused(array_index, debug=True)
        return False
    
    def test_attribute_access(self):
        """Attribute access not supported"""
        @gpu_kernel
        def attribute_error(a):
            return a.real  # Attribute access not supported
        
        compiled = self.math.compile_fused(attribute_error, debug=True)
        return False
    
    # -----------------------------------------------------------------
    # RUNTIME ERROR TESTS
    # -----------------------------------------------------------------
    
    def test_runtime_type_mismatch(self):
        """Runtime type mismatch (compile succeeds, runtime fails)"""
        @gpu_kernel({
            "a": (NP_GLTypes.float, "buffer"),
            "b": (NP_GLTypes.int, "buffer")
        })
        def runtime_type_error(a, b):
            return a + b  # Different types in operation
        
        try:
            compiled = self.math.compile_fused(runtime_type_error, debug=True)
            print("✓ Compilation succeeded (expected)")
            
            # Try to run with mismatched types
            x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
            y = np.array([4, 5, 6], dtype=np.int32)
            result = compiled(x, y)
            output = self.math.fetch(result)
            print(f"✓ Runtime execution succeeded: {output}")
            return True
        except Exception as e:
            print(f"✗ Runtime error: {type(e).__name__}: {e}")
            traceback.print_exc()
            return False
    
    # -----------------------------------------------------------------
    # COMPREHENSIVE TEST SUITE
    # -----------------------------------------------------------------
    
    def run_all_tests(self):
        """Run all positive and negative tests"""
        print("\n" + "="*80)
        print("GPU MATH COMPILATION ERROR HANDLING TEST SUITE")
        print("="*80)
        
        tests = [
            # Positive tests (should pass)
            ("POSITIVE: Basic Kernel", self.test_basic_kernel, True),
            ("POSITIVE: Typed Variables", self.test_typed_variables, True),
            ("POSITIVE: Inline Function", self.test_inline_function, True),
            ("POSITIVE: For Loop", self.test_for_loop, True),
            ("POSITIVE: If Statement", self.test_if_statement, True),
            ("POSITIVE: Nested Loops", self.test_nested_loops, True),
            ("POSITIVE: Built-in Functions", self.test_builtin_functions, True),
            
            # Negative tests (should fail with compilation errors)
            ("NEGATIVE: Undefined Variable", self.test_undefined_variable, False),
            ("NEGATIVE: Invalid For Loop", self.test_invalid_for_loop, False),
            ("NEGATIVE: Missing Return", self.test_missing_return, False),
            ("NEGATIVE: Unsupported Syntax", self.test_unsupported_syntax, False),
            ("NEGATIVE: Invalid Function Call", self.test_invalid_function_call, False),
            ("NEGATIVE: Complex Assignment", self.test_complex_assignment, False),
            ("NEGATIVE: While Loop Error", self.test_while_loop_error, False),
            ("NEGATIVE: Nested Function Error", self.test_nested_function_error, False),
            ("NEGATIVE: Array Indexing", self.test_array_indexing, False),
            ("NEGATIVE: Attribute Access", self.test_attribute_access, False),
            
            # # Runtime test
            # ("Runtime Type Mismatch", self.test_runtime_type_mismatch, True),
        ]
        
        results = []
        for name, test_func, should_pass in tests:
            passed = self.run_test(name, test_func, should_pass)
            results.append((name, passed, should_pass))
        
        # Summary
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        
        as_expected = sum([1 for name, passed, should_pass in results if passed])
        total_count = len(results)
        
        print(f"Total Tests: {total_count}")
        print(f"As Expected: {as_expected}")
        print(f"Failed: {total_count - as_expected}")
        
        print("\nDetailed Results:")
        for name, passed, should_pass in results:
            status = "✓ PASSED" if passed == should_pass else "✗ FAIL" 
            expectation = "✓ (as expected)" if passed else "✗ (UNEXPECTED RESULT)"
            print(f"  {name:30} {status:10} {expectation}")

        return as_expected == total_count

def main():
    """Run the error handling demo"""
    demo = ErrorHandlingDemo()
    success = demo.run_all_tests()
    demo.destroy()
    
    if success:
        print("\n✅ All tests completed successfully!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())