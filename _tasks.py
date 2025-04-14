import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent


def update_version_strings(file_path, new_version):
    # taken from:
    # https://stackoverflow.com/questions/57108712/replace-updated-version-strings-in-files-via-python
    version_regex = re.compile(r"(^_*?version_*?\s*=\s*\")(\d+\.\d+\.\d+-?\S*)\"", re.M)
    with open(file_path, "r+") as f:
        content = f.read()
        f.seek(0)
        f.write(
            re.sub(
                version_regex,
                lambda match: '{}{}"'.format(match.group(1), new_version),
                content,
            )
        )
        f.truncate()


def update_version(version):
    subprocess.run(["poetry", "version", version], shell=False, check=True)
    new_version = (
        subprocess.run(["poetry", "version"], shell=False, check=True, capture_output=True)
        .stdout.decode()
        .strip()
        .split(" ", 1)[1]
    )
    update_version_strings(HERE.joinpath("sleep-analysis/__init__.py"), new_version)


def task_update_version():
    update_version(sys.argv[1])


def task_new_experiment():
    name = sys.argv[1]
    files = ["__init__.py", "helper/__init__.py", "notebooks/.gitkeep", "scripts/.gitkeep", "README.md"]

    base_path: Path = HERE / "experiments" / name
    if base_path.is_file() or base_path.is_dir():
        raise ValueError(f"Experiment with name {name} already exists at {base_path}")

    for f in files:
        path = HERE / "experiments" / name / f
        path.parent.mkdir(exist_ok=True, parents=True)
        path.touch(exist_ok=True)
