import sys
import traceback
import inspect
import ast
from typing import Optional, Tuple, Any
from dataclasses import dataclass

@dataclass
class CompilationErrorInfo:
    """Container for compilation error information"""
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    column_offset: Optional[int] = None
    function_name: Optional[str] = None
    source_code: Optional[str] = None
    error_message: str = ""
    node_type: Optional[str] = None

class CompilationError(Exception):
    """Enhanced exception for compilation errors with source context"""
    def __init__(self, message: str, error_info: Optional[CompilationErrorInfo] = None):
        self.error_info = error_info or CompilationErrorInfo(error_message=message)
        if error_info:
            self.error_info.error_message = message
        super().__init__(self._format_message())
    
    def _format_message(self) -> str:
        """Format error message with context"""
        parts = ["Compilation Error:"]
        
        if self.error_info.file_path:
            parts.append(f"  File: {self.error_info.file_path}")
        
        if self.error_info.line_number:
            line_info = f"  Line: {self.error_info.line_number}"
            if self.error_info.column_offset:
                line_info += f", Column: {self.error_info.column_offset}"
            parts.append(line_info)
        
        if self.error_info.function_name:
            parts.append(f"  Function: {self.error_info.function_name}")
        
        if self.error_info.source_code:
            parts.append(f"  Source: {self.error_info.source_code.strip()}")
        
        if self.error_info.node_type:
            parts.append(f"  Node Type: {self.error_info.node_type}")
        
        parts.append(f"  Error: {self.error_info.error_message}")
        
        return "\n".join(parts)

def get_node_location(node: ast.AST, source: str) -> Tuple[int, int, str]:
    """
    Extract location information from an AST node
    
    Returns: (line_number, column_offset, source_line)
    """
    line_number = getattr(node, 'lineno', None)
    col_offset = getattr(node, 'col_offset', None)
    
    # Extract the source line
    source_line = ""
    if line_number and source:
        lines = source.split('\n')
        if 0 <= line_number - 1 < len(lines):
            source_line = lines[line_number - 1]
    
    return line_number, col_offset, source_line

def wrap_with_error_handling(func, source_code: str, function_name: str, 
                           file_path: Optional[str] = None):
    """
    Decorator to wrap function execution with error handling
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Get the current frame
            tb = traceback.extract_tb(sys.exc_info()[2])
            
            # Find the most relevant frame
            for frame in reversed(tb):
                if frame.name == func.__name__ or frame.name == function_name:
                    error_info = CompilationErrorInfo(
                        file_path=file_path or frame.filename,
                        line_number=frame.lineno,
                        column_offset=frame.colno if hasattr(frame, 'colno') else None,
                        function_name=function_name,
                        source_code=frame.line,
                        error_message=str(e)
                    )
                    raise CompilationError(str(e), error_info) from e
            
            # If no specific frame found, use default
            error_info = CompilationErrorInfo(
                file_path=file_path,
                function_name=function_name,
                error_message=str(e)
            )
            raise CompilationError(str(e), error_info) from e
    
    return wrapper

def create_error_context(node: ast.AST, source: str, function_name: str,
                        file_path: Optional[str] = None) -> CompilationErrorInfo:
    """
    Create error context from AST node
    """
    line_num, col_offset, source_line = get_node_location(node, source)
    
    return CompilationErrorInfo(
        file_path=file_path,
        line_number=line_num,
        column_offset=col_offset,
        function_name=function_name,
        source_code=source_line,
        node_type=node.__class__.__name__
    )

def print_compilation_error(error: CompilationError, show_traceback: bool = True):
    """
    Print compilation error with context
    """
    print("\n" + "="*60)
    print("COMPILATION ERROR DETAILS:")
    print("="*60)
    
    if error.error_info:
        print(str(error))
    
    if show_traceback:
        print("\nFull Traceback:")
        print("-"*40)
        traceback.print_exc()
    
    print("="*60)