import os
import sys
import importlib.util
from types import ModuleType
from typing import Dict, Optional, Any

def import_file(
    filepath: str, 
    engine_instance: Any = None,
    kungfu_module: Optional[ModuleType] = None,
    extra_modules: Optional[Dict[str, Any]] = None,
    module_name: Optional[str] = None
) -> ModuleType:
    # Validate filepath
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    if not os.path.isfile(filepath):
        raise ValueError(f"Path is not a file: {filepath}")
    
    # Generate module name if not provided
    if module_name is None:
        # Create a unique module name based on the filepath
        base_name = os.path.basename(filepath).replace('.py', '').replace('.', '_')
        abs_path = os.path.abspath(filepath)
        # Use hash to ensure uniqueness
        path_hash = abs(hash(abs_path)) % 10000
        module_name = f"_kungfu_import_{base_name}_{path_hash}"
    
    # Create module spec from file location
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create module spec for: {filepath}")
    
    # Create the module from spec
    module = importlib.util.module_from_spec(spec)
    
    # Prepare injected modules dictionary
    injected = {}
    
    # Inject engine if provided
    if engine_instance is not None:
        injected['engine'] = engine_instance
    
    # Inject kungfu if provided
    if kungfu_module is not None:
        injected['kungfu'] = kungfu_module
        # Also make it available as 'kf' for convenience
        injected['kf'] = kungfu_module
    
    # Add any extra modules
    if extra_modules:
        injected.update(extra_modules)
    
    # Add injected modules to sys.modules so imports work
    original_modules = {}
    for name, obj in injected.items():
        # Save original if it exists
        if name in sys.modules:
            original_modules[name] = sys.modules[name]
        # Add to sys.modules
        sys.modules[name] = obj
    
    # Add to sys.modules to make imports work properly
    sys.modules[module_name] = module
    
    # Execute the module
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        # Clean up sys.modules on failure
        if module_name in sys.modules:
            del sys.modules[module_name]
        # Restore original modules
        for name in injected.keys():
            if name in original_modules:
                sys.modules[name] = original_modules[name]
            elif name in sys.modules:
                del sys.modules[name]
        raise ImportError(f"Error executing module {filepath}: {e}") from e
    
    # Clean up injected modules from sys.modules after execution
    # (they're still available in the module's __dict__)
    for name in injected.keys():
        if name in sys.modules and sys.modules[name] is injected[name]:
            if name in original_modules:
                # Restore original
                sys.modules[name] = original_modules[name]
            else:
                # Remove if it wasn't there before
                del sys.modules[name]
    
    # But also keep them in the module's __dict__ for direct access
    for name, obj in injected.items():
        module.__dict__[name] = obj
    
    return module


def import_file_simple(filepath: str, **kwargs) -> ModuleType:
    return import_file(
        filepath=filepath,
        engine_instance=kwargs.pop('engine', None),
        kungfu_module=kwargs.pop('kungfu', None),
        extra_modules=kwargs if kwargs else None
    )


class ModuleLoader:
    def __init__(self, engine_instance: Any, kungfu_module: Optional[ModuleType] = None):
        self.engine = engine_instance
        self.kungfu = kungfu_module
        self.loaded_modules: Dict[str, ModuleType] = {}
    
    def import_file(
        self, 
        filepath: str, 
        extra_modules: Optional[Dict[str, Any]] = None,
        module_name: Optional[str] = None
    ) -> ModuleType:
        module = import_file(
            filepath=filepath,
            engine_instance=self.engine,
            kungfu_module=self.kungfu,
            extra_modules=extra_modules,
            module_name=module_name
        )

        abs_path = os.path.abspath(filepath)
        self.loaded_modules[abs_path] = module
        
        return module
    
    def reload(self, filepath: str) -> ModuleType:
        return self.import_file(filepath)
    
    def get_module(self, filepath: str) -> Optional[ModuleType]:
        abs_path = os.path.abspath(filepath)
        return self.loaded_modules.get(abs_path)