"""
Unit and regression test for the simpack package.
"""

# Import package, test suite, and other packages as needed
import simpack
import pytest
import sys

def test_simpack_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "simpack" in sys.modules
