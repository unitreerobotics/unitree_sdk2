#!/usr/bin/env python3
"""Test script to verify unitree_sdk2 wheel functionality."""

import sys
import importlib.metadata

def test_import():
    """Test basic import."""
    try:
        import unitree_interface
        print(f"✓ Import successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_version():
    """Test version info."""
    try:
        version = importlib.metadata.version('unitree_sdk2')
        print(f"✓ Version: {version}")
        return True
    except Exception as e:
        print(f"✗ Version check failed: {e}")
        return False

def test_attributes():
    """Test basic attributes."""
    try:
        import unitree_interface
        attrs = ['RobotType', 'create_robot']
        for attr in attrs:
            if hasattr(unitree_interface, attr):
                print(f"✓ Has attribute: {attr}")
            else:
                print(f"✗ Missing attribute: {attr}")
                return False
        return True
    except Exception as e:
        print(f"✗ Attribute check failed: {e}")
        return False

def main():
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print("-" * 60)
    
    results = []
    results.append(test_import())
    results.append(test_version())
    results.append(test_attributes())
    
    print("-" * 60)
    if all(results):
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
