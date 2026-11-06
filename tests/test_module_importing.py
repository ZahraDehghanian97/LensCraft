"""Third-party namespace isolation, requiring only the Python standard library."""

import importlib
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest

from utils.importing import ModuleImporter


class ModuleImporterTests(unittest.TestCase):
    PREFIXES = ('src', 'clatr', 'utils', '_module_import_fixture')

    @staticmethod
    def selected(name):
        return any(name == prefix or name.startswith(prefix + '.')
                   for prefix in ModuleImporterTests.PREFIXES)

    def setUp(self):
        self.paths = sys.path.copy()
        self.cwd = os.getcwd()
        self.modules = {name: module for name, module in sys.modules.copy().items() if self.selected(name)}
        self.attributes = [(module, vars(module).copy()) for module in self.modules.values()
                           if isinstance(module, ModuleType)]
        for name in list(sys.modules):
            if name in ('src', 'clatr') or name.startswith(('src.', 'clatr.')):
                del sys.modules[name]
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.host = self.root / 'host'
        self.external = self.root / 'director'
        self.write(self.host, 'src/__init__.py', "ORIGIN = 'host'\n")
        self.write(self.host, 'src/host_only.py', "ORIGIN = 'host_child'\n")
        # The real DIRECTOR directories are namespace packages without init.
        self.write(self.external, 'src/datasets.py', "ORIGIN = 'director'\nclass Dataset:\n    def origin(self):\n        return ORIGIN\n")
        self.write(self.external, 'src/late.py', "ORIGIN = 'late_director'\n")
        self.write(self.external, 'clatr/src/losses.py', "ORIGIN = 'clatr'\n")
        sys.path.insert(0, str(self.host))

    def tearDown(self):
        os.chdir(self.cwd)
        sys.path[:] = self.paths
        for name in list(sys.modules):
            if self.selected(name):
                del sys.modules[name]
        sys.modules.update(self.modules)
        for module, attributes in self.attributes:
            vars(module).clear()
            vars(module).update(attributes)
        self.temporary.cleanup()
        importlib.invalidate_caches()

    @staticmethod
    def write(root, relative, source):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
        return path

    def test_namespace_wins_over_later_regular_src_and_supports_late_imports(self):
        paths = sys.path.copy()
        with ModuleImporter.temporary_module(self.external):
            module = importlib.import_module('src.datasets')
            self.assertEqual(module.ORIGIN, 'director')
            self.assertEqual(importlib.import_module('clatr.src.losses').ORIGIN, 'clatr')
            self.assertEqual(sys.path[1:], paths)
        self.assertEqual(sys.path, paths)
        self.assertIs(importlib.import_module('src.datasets'), module)
        self.assertEqual(importlib.import_module('src.late').ORIGIN, 'late_director')

    def test_cached_host_src_graph_restored_and_returned_objects_still_work(self):
        host = importlib.import_module('src')
        child = importlib.import_module('src.host_only')
        original = ModuleImporter._module_graph('src')
        with ModuleImporter.temporary_module(self.external):
            self.assertIsNot(sys.modules['src'], host)
            self.assertNotIn('src.host_only', sys.modules)
            dataset = importlib.import_module('src.datasets').Dataset()
            self.assertEqual(dataset.origin(), 'director')
        self.assertEqual(ModuleImporter._module_graph('src'), original)
        self.assertIs(host.host_only, child)
        self.assertFalse(hasattr(host, 'datasets'))
        self.assertEqual(dataset.origin(), 'director')

    def test_same_third_party_graph_reused_on_repeated_contexts(self):
        with ModuleImporter.temporary_module(self.external):
            first = importlib.import_module('src.datasets')
        with ModuleImporter.temporary_module(self.external):
            second = importlib.import_module('src.datasets')
            late = importlib.import_module('src.late')
        self.assertIs(first, second)
        self.assertIs(importlib.import_module('src.late'), late)

    def test_nested_different_namespace_restores_outer_graph(self):
        alternative = self.root / 'other_director'
        self.write(alternative, 'src/datasets.py', "ORIGIN = 'other'\n")
        with ModuleImporter.temporary_module(self.external):
            first = importlib.import_module('src.datasets')
            with ModuleImporter.temporary_module(alternative):
                second = importlib.import_module('src.datasets')
                self.assertEqual(second.ORIGIN, 'other')
            self.assertIs(importlib.import_module('src.datasets'), first)
        self.assertIs(importlib.import_module('src.datasets'), first)

    def test_replaced_utils_restore_both_registry_and_parent_attributes(self):
        utils = importlib.import_module('utils')
        name = 'utils._module_import_fixture'
        original = ModuleType(name)
        original.ORIGIN = 'host_utility'
        sys.modules[name] = original
        utils._module_import_fixture = original
        self.write(self.external, 'utils/_module_import_fixture.py', "ORIGIN = 'external_utility'\n")
        with ModuleImporter.temporary_module(self.external, [name]):
            from utils import _module_import_fixture
            self.assertIs(_module_import_fixture, sys.modules[name])
            self.assertEqual(_module_import_fixture.ORIGIN, 'external_utility')
        self.assertIs(sys.modules[name], original)
        self.assertIs(utils._module_import_fixture, original)

    def test_new_replaced_utils_removed_after_context(self):
        utils = importlib.import_module('utils')
        name = 'utils._module_import_fixture'
        self.write(self.external, 'utils/_module_import_fixture.py', "VALUE = 42\n")
        self.assertNotIn(name, sys.modules)
        self.assertFalse(hasattr(utils, '_module_import_fixture'))
        with ModuleImporter.temporary_module(self.external, [name]):
            self.assertEqual(importlib.import_module(name).VALUE, 42)
        self.assertNotIn(name, sys.modules)
        self.assertFalse(hasattr(utils, '_module_import_fixture'))

    def test_failed_setup_restores_cwd_paths_utilities_and_host_namespace(self):
        host = importlib.import_module('src')
        importlib.import_module('src.host_only')
        original = ModuleImporter._module_graph('src')
        utils = importlib.import_module('utils')
        name = 'utils._module_import_fixture'
        saved = ModuleType(name)
        sys.modules[name] = saved
        utils._module_import_fixture = saved
        self.write(self.external, 'utils/_module_import_fixture.py', "VALUE = 'temporary'\n")
        self.write(self.external, 'utils/_broken_import_fixture.py',
                   "from src import datasets\nraise RuntimeError('setup failed')\n")
        paths = sys.path.copy()
        with self.assertRaisesRegex(RuntimeError, 'setup failed'):
            with ModuleImporter.temporary_module(
                self.external, [name, 'utils._broken_import_fixture'], self.external,
            ):
                self.fail('Context must not yield after failed setup')
        self.assertEqual(os.getcwd(), self.cwd)
        self.assertEqual(sys.path, paths)
        self.assertEqual(ModuleImporter._module_graph('src'), original)
        self.assertIs(sys.modules['src'], host)
        self.assertIs(sys.modules[name], saved)
        self.assertIs(utils._module_import_fixture, saved)
        self.assertNotIn('utils._broken_import_fixture', sys.modules)
        self.assertFalse(hasattr(utils, '_broken_import_fixture'))
        self.assertNotIn('clatr', sys.modules)

    def test_body_failure_rolls_back_new_imports_in_reused_namespace(self):
        with ModuleImporter.temporary_module(self.external):
            original = importlib.import_module('src.datasets')
        package = sys.modules['src']
        with self.assertRaisesRegex(RuntimeError, 'body failed'):
            with ModuleImporter.temporary_module(self.external):
                importlib.import_module('src.late')
                raise RuntimeError('body failed')
        self.assertIs(importlib.import_module('src.datasets'), original)
        self.assertNotIn('src.late', sys.modules)
        self.assertFalse(hasattr(package, 'late'))

    def test_file_import_registers_identity_before_dataclass_execution(self):
        path = self.write(self.external, 'example.py',
                          "from dataclasses import dataclass\n"
                          "@dataclass\nclass Example:\n    value: 'int' = 3\n")
        module = ModuleImporter.import_module('_module_import_fixture', path)
        self.assertEqual(module.Example().value, 3)


if __name__ == '__main__':
    unittest.main()
