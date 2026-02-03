"""
Run all edge deployment tests.

Usage:
    python tests/run_all_tests.py
"""
import os
import sys
import subprocess

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))


def run_test(name, script):
    """Run a test script and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print("=" * 60)
    
    script_path = os.path.join(TESTS_DIR, script)
    
    if not os.path.exists(script_path):
        print(f"  ERROR: Script not found: {script_path}")
        return False
    
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=False,
    )
    
    return result.returncode == 0


def main():
    print("=" * 60)
    print("Edge Deployment Test Suite")
    print("=" * 60)
    
    tests = [
        ("BiomedCLIP INT8", "test_biomedclip.py"),
        ("MedGemma Q4_K_S", "test_medgemma.py"),
    ]
    
    results = {}
    
    for name, script in tests:
        try:
            results[name] = run_test(name, script)
        except Exception as e:
            print(f"  ERROR: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUITE SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("  ALL TESTS PASSED")
        return 0
    else:
        print("  SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
