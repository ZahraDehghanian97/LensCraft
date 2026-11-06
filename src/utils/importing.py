from contextlib import contextmanager
from importlib.machinery import ModuleSpec
from importlib.util import module_from_spec, spec_from_file_location
import os
import sys
import logging
from types import ModuleType

logger = logging.getLogger(__name__)
_MISSING = object()

class ModuleImporter:
    @staticmethod
    def import_module(module_name, path):
        path = os.fspath(path)
        if os.path.isdir(path):
            ModuleImporter._bind_package(module_name, path)
            py_files = sorted(f for f in os.listdir(path) if f.endswith('.py') and f != '__init__.py')
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
        full_path = os.fspath(full_path)
        if not full_path.endswith('.py'):
            full_path = full_path + '.py'
        
        spec = spec_from_file_location(module_name, full_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to create module spec for {full_path}")

        module = module_from_spec(spec)
        previous = sys.modules.get(module_name, _MISSING)
        # Imports executed by the module itself (and dataclass decorators)
        # require its identity to exist before its body runs.
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            if previous is _MISSING:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = previous
            raise
        parent, _, child = module_name.rpartition('.')
        if parent in sys.modules:
            setattr(sys.modules[parent], child, module)
        return module

    @staticmethod
    def _bind_package(name, directory):
        initializer = os.path.join(directory, '__init__.py')
        if os.path.isfile(initializer):
            return ModuleImporter._import_file(name, initializer)
        # DIRECTOR's src/clatr folders have no __init__.py. Merely putting
        # their root first on sys.path loses to LensCraft's regular src package
        # later on sys.path, even when src has not been imported yet. Bind an
        # explicit search path so Python cannot substitute that other package.
        spec = ModuleSpec(name, loader=None, is_package=True)
        spec.submodule_search_locations = [directory]
        package = module_from_spec(spec)
        package.__path__ = [directory]
        sys.modules[name] = package
        return package

    @staticmethod
    def _module_graph(name):
        return {key: value for key, value in sys.modules.copy().items()
                if key == name or key.startswith(name + '.')}

    @staticmethod
    def _restore_graph(name, original, attributes):
        current = ModuleImporter._module_graph(name)
        for key in current:
            sys.modules.pop(key, None)
        sys.modules.update(original)
        # Importing a submodule also adds an attribute to its parent. Restore
        # these links along with sys.modules, including failed nested imports.
        for package, saved in attributes:
            for key, value in list(vars(package).items()):
                if isinstance(value, ModuleType) and (
                    value.__name__ == name or value.__name__.startswith(name + '.')
                ) and key not in saved:
                    delattr(package, key)
            for key, value in saved.items():
                setattr(package, key, value)

    @staticmethod
    @contextmanager
    def temporary_module(path, replace_modules=None, working_dir=None):
        """Temporarily expose DIRECTOR imports without replacing host packages.

        Only src/clatr package graphs and explicitly requested utility modules
        are isolated. Existing host graphs are restored. With no conflicting
        host graph, successfully loaded third-party packages stay available for
        later imports and pickle/class resolution, matching the old behavior.
        Returned objects keep their imported classes and function globals when
        a conflicting host graph has to be restored.
        """
        path = os.path.abspath(os.fspath(path))
        replace_modules = tuple(replace_modules or ())
        original_sys_path = sys.path.copy()
        original_cwd = os.getcwd() if working_dir is not None else None
        original_modules = {name: sys.modules.get(name, _MISSING) for name in replace_modules}
        original_attributes = []
        for name in replace_modules:
            parent, _, child = name.rpartition('.')
            package = sys.modules.get(parent)
            if package is not None:
                original_attributes.append((package, child, getattr(package, child, _MISSING)))
        namespaces = []
        completed = False
        try:
            # Keep every original path; callers may still need LensCraft's
            # distinct top-level packages (models/data/training/utils).
            sys.path.insert(0, path)
            if working_dir is not None:
                logger.info(f"Changing working directory from {original_cwd} to {working_dir}")
                os.chdir(working_dir)
            for name in ('src', 'clatr'):
                directory = os.path.join(path, name)
                if not os.path.isdir(directory):
                    continue
                saved = ModuleImporter._module_graph(name)
                attributes = [
                    (module, {key: value for key, value in vars(module).items()
                              if isinstance(value, ModuleType) and (
                                  value.__name__ == name or value.__name__.startswith(name + '.')
                              )})
                    for module in saved.values() if isinstance(module, ModuleType)
                ]
                root = saved.get(name)
                locations = list(getattr(root, '__path__', ()))
                same_package = bool(locations) and all(
                    os.path.realpath(location) == os.path.realpath(directory) for location in locations
                )
                namespaces.append((name, saved, attributes, same_package))
                if not same_package:
                    for key in saved:
                        sys.modules.pop(key, None)
                    ModuleImporter._bind_package(name, directory)
            for name in replace_modules:
                sys.modules.pop(name, None)
                ModuleImporter.import_module(name, os.path.join(path, name.replace('.', os.path.sep)))
            yield
            completed = True
        finally:
            for name, saved, attributes, same_package in reversed(namespaces):
                if not completed or (saved and not same_package):
                    ModuleImporter._restore_graph(name, saved, attributes)
            for name, module in original_modules.items():
                if module is _MISSING:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module
            for package, child, original in original_attributes:
                if original is _MISSING:
                    if hasattr(package, child):
                        delattr(package, child)
                else:
                    setattr(package, child, original)
            sys.path[:] = original_sys_path
            if original_cwd is not None:
                logger.info(f"Restoring working directory from {os.getcwd()} to {original_cwd}")
                os.chdir(original_cwd)
