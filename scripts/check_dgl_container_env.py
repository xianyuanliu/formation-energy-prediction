import importlib

modules = [
    "torch",
    "dgl",
    "matgl",
    "alignn",
    "jarvis",
    "pymatgen",
    "ase",
    "sklearn",
    "pandas",
    "numpy",
    "scipy",
]

for name in modules:
    module = importlib.import_module(name)
    print(f"{name}: {getattr(module, '__version__', 'imported')}")

import torch

print(f"torch cuda: {torch.version.cuda}")
print(f"cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu count: {torch.cuda.device_count()}")
    print(f"gpu 0: {torch.cuda.get_device_name(0)}")

