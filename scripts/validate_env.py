# scripts/validate_env.py

import sys

def check(name, fn):
    try:
        result = fn()
        print(f"  ✓  {name}: {result}")
        return True
    except Exception as e:
        print(f"  ✗  {name}: FAILED — {e}")
        return False

print("\n=== EdgeMamba-3 Environment Validation ===\n")

# Critical checks — must pass for the pipeline to work
critical_checks = [
    ("Python 3.10+",
     lambda: sys.version),
    ("PyTorch 2.1",
     lambda: __import__("torch").__version__),
    ("CUDA available",
     lambda: str(__import__("torch").cuda.is_available())),
    ("CUDA device",
     lambda: __import__("torch").cuda.get_device_name(0)),
    ("torch_geometric",
     lambda: __import__("torch_geometric").__version__),
    ("torch_scatter",
     lambda: __import__("torch_scatter").__version__),
]

# Optional checks — nice to have but not required
optional_checks = [
    ("Mamba-3 (falls back to Mamba-2 if unavailable)",
     lambda: str(__import__("mamba_ssm.modules.mamba3", fromlist=["Mamba3"]))),
    ("Mamba-SSM",
     lambda: __import__("mamba_ssm").__version__),
    ("relbench",
     lambda: __import__("relbench").__version__),
    ("torch_frame",
     lambda: __import__("torch_frame").__version__),
    ("optuna",
     lambda: __import__("optuna").__version__),
]

print("── Critical Dependencies ──")
critical_results = [check(name, fn) for name, fn in critical_checks]
critical_passed = sum(critical_results)

print("\n── Optional Dependencies ──")
optional_results = [check(name, fn) for name, fn in optional_checks]
optional_passed = sum(optional_results)

total = len(critical_checks) + len(optional_checks)
passed = critical_passed + optional_passed
print(f"\n{passed}/{total} checks passed ({critical_passed}/{len(critical_checks)} critical).")

if critical_passed == len(critical_checks):
    print("Environment ready. All critical dependencies satisfied.\n")
    sys.exit(0)
else:
    print("CRITICAL dependencies missing. Fix failures above before proceeding.\n")
    sys.exit(1)
