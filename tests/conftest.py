def pytest_configure(config):
    # importlib mode stops pytest from importing the parent package
    # __init__.py, whose relative imports only work under ComfyUI's
    # node loader.
    config.option.importmode = "importlib"

