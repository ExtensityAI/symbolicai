import pytest
import subprocess

def test_local_install():
    subprocess.run(['sympkg', 'r', 'ExstensityAI-gat/diff-symai'], stdout=subprocess.PIPE)

    result = subprocess.run(
        ['sympkg', 'i', 'ExstensityAI-gat/diff-symai', '/home/x000ff4/open_source/valko/ExstensityAI-gat'],
        stdout=subprocess.PIPE)
    assert b'Using local' in result.stdout

def test_github_install():
    subprocess.run(['sympkg', 'r', 'ExstensityAI-gat/diff-symai'], stdout=subprocess.PIPE)

    result = subprocess.run(
        ['sympkg', 'i', 'ExstensityAI-gat/diff-symai'],
        stdout=subprocess.PIPE)
    assert b'Using github' in result.stdout

if __name__ == "__main__":
    pytest.main()
