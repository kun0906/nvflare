
def check_imported_libraries() -> None:
    import sys
    import os

    def get_module_paths():
        module_paths = {}
        for module_name, module in sys.modules.items():
            if module is not None and hasattr(module, '__file__'):
                try:
                    file_path = os.path.abspath(module.__file__)
                    module_paths[module_name] = file_path
                except Exception as e:
                    print(f"Could not determine the path for module {module_name}: {e}")
        return module_paths

    # Print paths of all imported modules
    module_paths = get_module_paths()
    for module_name, file_path in module_paths.items():
        print(f"{module_name}: {file_path}")


check_imported_libraries()