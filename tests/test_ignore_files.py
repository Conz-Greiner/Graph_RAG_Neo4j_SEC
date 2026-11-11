import os
import pytest

def test_gitignore_exists_and_contains_entries():
    assert os.path.exists(".gitignore")
    with open(".gitignore", "r") as f:
        content = f.read()
    assert ".env" in content
    assert "data/" in content
    assert "logs/" in content
    assert "__pycache__/" in content

def test_dockerignore_exists_and_contains_entries():
    assert os.path.exists(".dockerignore")
    with open(".dockerignore", "r") as f:
        content = f.read()
    assert ".env" in content
    assert ".git" in content
    assert "data/" in content
    assert "logs/" in content

def test_dockerignore_does_not_exclude_src_or_config():
    with open(".dockerignore", "r") as f:
        content = f.read()
    lines = [line.strip() for line in content.splitlines() if line.strip() and not line.startswith('#')]
    assert "src/" not in lines
    assert "config/" not in lines
    assert "src" not in lines
    assert "config" not in lines