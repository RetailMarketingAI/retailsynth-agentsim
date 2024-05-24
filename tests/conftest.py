import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

pytest_plugins = [
    "tests.fixtures.env_fixture_test",
]
