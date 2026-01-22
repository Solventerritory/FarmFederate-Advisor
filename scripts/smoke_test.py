"""Smoke test runner for FarmFederate
Runs a quick smoke test by invoking the main Colab script with --smoke-test flag.
Exits nonzero on failure.
"""
import subprocess
import sys
import shutil

PY = sys.executable
SCRIPT = "FarmFederate_Complete_Colab.py"

try:
    res = subprocess.run([PY, SCRIPT, "--smoke-test"], check=True)
    print("Smoke test completed successfully")
    sys.exit(0)
except subprocess.CalledProcessError as e:
    print("Smoke test failed with return code", e.returncode)
    sys.exit(e.returncode)
except Exception as e:
    print("Smoke test runner error:", e)
    sys.exit(2)
