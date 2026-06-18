import subprocess


def run(cmd):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=False)


if __name__ == "__main__":
    run("ruff check . --fix --unsafe-fixes")
    run("ruff format .")
    run("ruff check .")
