import torch

from comfy.cli_args import args

# Must precede the import: comfy.model_management picks its device at import
# time, and a CUDA build with no driver raises there.
if not torch.cuda.is_available():
    args.cpu = True


def pytest_configure(config):
    # importlib mode stops pytest from importing the parent package
    # __init__.py, whose relative imports only work under ComfyUI's
    # node loader.
    config.option.importmode = "importlib"

