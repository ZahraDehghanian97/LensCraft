from contextlib import contextmanager
from importlib.util import module_from_spec, spec_from_file_location
import os
import sys


class ModuleImporter:
    @staticmethod
    def import_module(module_name, path):
        if os.path.isdir(path):
            py_files = [f for f in os.listdir(path) if f.endswith('.py')]
            for py_file in py_files:
                file_name = os.path.splitext(py_file)[0]
                full_module_name = f"{module_name}.{file_name}"
                full_path = os.path.join(path, py_file)
                ModuleImporter._import_file(full_module_name, full_path)
            return sys.modules[module_name]
        else:
            return ModuleImporter._import_file(module_name, path)

    @staticmethod
    def _import_file(module_name, full_path):
        spec = spec_from_file_location(module_name, full_path + '.py')
        if spec is None:
            raise ImportError(f"Failed to create module spec for {full_path}")

        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[module_name] = module
        return module

    @staticmethod
    @contextmanager
    def temporary_module(path, replace_modules=[]):
        original_sys_path = sys.path.copy()
        original_modules = {name: sys.modules[name]
                            for name in replace_modules if name in sys.modules}

        for module_name in replace_modules:
            if module_name in sys.modules:
                del sys.modules[module_name]

            ModuleImporter.import_module(module_name, os.path.join(
                path, module_name.replace('.', os.path.sep)))

        sys.path.insert(0, path)
        try:
            yield
        finally:
            sys.path = original_sys_path
            for name, module in original_modules.items():
                sys.modules[name] = module
            for name in replace_modules:
                if name not in original_modules and name in sys.modules:
                    del sys.modules[name]
