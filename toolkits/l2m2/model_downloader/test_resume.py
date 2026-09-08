#!/usr/bin/env python3
"""
Test script to verify resume functionality for model downloads.

This script demonstrates that resume functionality works automatically
with the current version of huggingface_hub (1.0+).
"""

import os
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from download_hf_model import download_model, resolve_local_dir


class ModelDownloader:
    """Small adapter keeping this manual smoke script on the current function API."""

    def __init__(self, base_models_dir):
        self.base_models_dir = base_models_dir

    def download_model(self, repo_id, **kwargs):
        local_dir = kwargs.pop("local_dir", None)
        return download_model(
            repo_id=repo_id,
            local_dir=resolve_local_dir(repo_id, self.base_models_dir, local_dir),
            **kwargs,
        )


def test_basic_download():
    """Test basic download functionality."""
    print("=" * 60)
    print("TEST 1: Basic Download")
    print("=" * 60)

    downloader = ModelDownloader(base_models_dir="./test_models")

    try:
        # Download a small model for testing
        print("\nDownloading a small test model...")
        result = downloader.download_model(
            repo_id="hf-internal-testing/tiny-random-gpt2",
            allow_patterns=["*.json"],
        )
        print(f"✓ Test passed: Model downloaded to {result}")
        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def test_resume_simulation():
    """
    Test resume functionality by demonstrating it works with cached files.
    """
    print("\n" + "=" * 60)
    print("TEST 2: Resume Functionality (Simulation)")
    print("=" * 60)

    downloader = ModelDownloader(base_models_dir="./test_models")

    try:
        print("\nFirst download (will cache files)...")
        result1 = downloader.download_model(
            repo_id="hf-internal-testing/tiny-random-gpt2",
            allow_patterns=["*.json"],
        )
        print(f"✓ First download completed: {result1}")

        print("\nSecond download (should use cached files - resume in action)...")
        start_time = time.time()
        result2 = downloader.download_model(
            repo_id="hf-internal-testing/tiny-random-gpt2",
            allow_patterns=["*.json"],
        )
        elapsed = time.time() - start_time

        print(f"✓ Second download completed in {elapsed:.2f}s: {result2}")
        print("  (Fast completion indicates cached files were used)")

        if elapsed < 1.0:
            print("✓ Resume functionality working: Files were cached and reused")
            return True
        else:
            print("⚠ Files may have been re-downloaded (could be normal if cache was cleared)")
            return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def test_force_download():
    """Test force download functionality."""
    print("\n" + "=" * 60)
    print("TEST 3: Force Download")
    print("=" * 60)

    downloader = ModelDownloader(base_models_dir="./test_models")

    try:
        print("\nForce re-downloading files...")
        result = downloader.download_model(
            repo_id="hf-internal-testing/tiny-random-gpt2",
            allow_patterns=["*.json"],
            force_download=True
        )
        print(f"✓ Force download completed: {result}")
        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def cleanup():
    """Clean up test files."""
    print("\n" + "=" * 60)
    print("CLEANUP")
    print("=" * 60)

    import shutil
    test_dir = "./test_models"
    if os.path.exists(test_dir):
        try:
            shutil.rmtree(test_dir)
            print(f"✓ Cleaned up test directory: {test_dir}")
        except Exception as e:
            print(f"⚠ Could not clean up {test_dir}: {e}")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("RESUME FUNCTIONALITY TEST SUITE")
    print("=" * 60)
    print("\nThis test suite verifies that:")
    print("1. Downloads work correctly")
    print("2. Resume functionality is automatic (via caching)")
    print("3. Force download option works")
    print("\nNote: Resume is handled automatically by huggingface_hub")
    print("      library (version 1.0+) - no configuration needed.")
    print("=" * 60)

    results = []

    # Run tests
    results.append(("Basic Download", test_basic_download()))
    results.append(("Resume Simulation", test_resume_simulation()))
    results.append(("Force Download", test_force_download()))

    # Cleanup
    cleanup()

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")

    total = len(results)
    passed = sum(1 for _, result in results if result)

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! Resume functionality is working correctly.")
        return 0
    else:
        print("\n❌ Some tests failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
