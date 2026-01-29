import os
import sys
import importlib.util
from types import ModuleType
from typing import Dict
from direct.showbase.ShowBase import ShowBase
from panda3d.core import load_prc_file_data
import kungfu

_0_10 = tuple(str(i) for i in range(10))
TEST_FOLDER = os.path.join(os.path.dirname(__file__), "tests/tests")


def load_test_from_file(test_file, injected_modules: Dict[str, ModuleType]):
    module_name = f"_test_{os.path.basename(test_file).replace('.','_')}"

    spec = importlib.util.spec_from_file_location(module_name, test_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec for {test_file}")

    module = importlib.util.module_from_spec(spec)

    for name, mod in injected_modules.items():
        module.__dict__[name] = mod

    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    if not hasattr(module, "test") or not callable(module.test):
        raise AttributeError("Test module must define callable `test()`")

    return module


class Test:
    def __init__(self, test_file: os.PathLike):
        self.name        = os.path.basename(os.path.dirname(test_file)).removesuffix(".py")
        self.test_file   = test_file
        self.module      = None
        self.loaded      = False
        self.test_passed = None
        self.error       = None

    def run(self, engine: kungfu.GPUMath, injected_modules):
        try:
            self.module = load_test_from_file(self.test_file, injected_modules)
            self.loaded = True
        except Exception as e:
            self.loaded = False
            self.error = e
            return False

        try:
            self.test_passed = bool(self.module.test(engine))
            return self.test_passed
        except Exception as e:
            self.test_passed = False
            self.error = e
            return False


class Tester:
    def __init__(self, app: ShowBase, test_folder: os.PathLike, injected_modules=None):
        self.app = app
        self.test_folder = test_folder
        self.injected_modules = injected_modules or {}
        
        self.found_tests = {}
        self.failed_load = {}
        self.failed_run  = {}
        self.failed_test = {}
        self.successful  = {}

    def _load_tests(self):
        to_search = []
        for ent in os.scandir(self.test_folder):
            if ent.is_dir() and ent.name.startswith(_0_10):
                to_search.append(ent.path)

        to_load = []
        for folder in to_search:
            for ent in os.scandir(folder):
                if ent.is_file() and ent.name == "test.py":
                    to_load.append(ent.path)
        print("to_load", to_load)
        for test_file in to_load:
            t = Test(test_file)
            print(t, t.name)
            self.found_tests[t.name] = t

    def run_all(self):
        self._load_tests()

        for name, test in self.found_tests.items():
            result = test.run(self.app.engine, self.injected_modules)

            if not test.loaded:
                self.failed_load[name] = test
            elif test.test_passed is False:
                self.failed_test[name] = test
            else:
                self.successful[name] = test

            self.app.task_mgr.step()

    def summary(self):
        print("\n=== TEST SUMMARY ===")
        print(f"Total:      {len(self.found_tests)}")
        print(f"Passed:     {len(self.successful)}")
        print(f"Failed:     {len(self.failed_test)} - {self.failed_test}")
        print(f"Load Error: {len(self.failed_load)}")

        if self.failed_test or self.failed_load:
            print("\n--- FAILURES ---")
            for name, t in {**self.failed_test, **self.failed_load}.items():
                print(f"{name}: {t.error}")


class TestApp(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)
        self.engine = kungfu.GPUMath(self, headless=True)

if __name__ == "__main__":
    for k, v in {
        "window-type": "none",
        "audio-library-name": "null",
        "sync-video": "#f",
    }.items():
        load_prc_file_data("", f"{k} {v}")

    app = TestApp()
    app.task_mgr.step()
    t = Tester(
        app,
        os.path.join(TEST_FOLDER, "compute"),
        injected_modules={"kungfu":kungfu}
    )
    t.run_all()
    t.summary()