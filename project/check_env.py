#!/usr/bin/env python3
"""Quick environment check. Run this first: `python3 check_env.py`"""
import shutil
import sys

REQUIRED = ["numpy", "matplotlib", "torch"]
OPTIONAL_BIN = ["pdflatex"]

def main():
    ok = True
    print(f"Python: {sys.version.split()[0]}\n")

    print("Required Python packages:")
    for pkg in REQUIRED:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "unknown")
            print(f"  [OK]   {pkg:<12} {ver}")
        except ImportError:
            print(f"  [MISS] {pkg:<12} -- run: pip install -r requirements.txt")
            ok = False

    print("\nOptional tools (only needed to recompile .tex docs; PDFs are already built):")
    for b in OPTIONAL_BIN:
        path = shutil.which(b)
        print(f"  [{'OK' if path else 'MISS'}]   {b:<12} {path or '(not found -- skip .tex recompilation)'}")

    print()
    if ok:
        print("Environment looks good. See README.md for what to run.")
    else:
        print("Missing required packages -- install them before running the demos.")
        sys.exit(1)

if __name__ == "__main__":
    main()
