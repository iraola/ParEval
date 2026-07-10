import argparse
import os
import re

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", default="kernel", help="Directory to store generated pycompss files (default: kernel)")
args = parser.parse_args()

# Root directory containing your 12 folders
root_dir = args.output_dir

for subdir, _, files in os.walk(root_dir):
    if "pycompss" in files:
        pycompss_path = os.path.join(subdir, "pycompss")
        serial_python_path = os.path.join(subdir, "serial-python")

        with open(pycompss_path, "r") as f:
            content = f.read()

        # Remove line that contains the programming model name
        lines = content.splitlines()
        matches = [line for line in lines if "PyCOMPSs" in line]

        if len(matches) == 0:
            raise ValueError(f"Could not find 'PyCOMPSs' in {pycompss_path}")
        if len(matches) > 1:
            raise ValueError(f"Found multiple occurrences of 'PyCOMPSs' in {pycompss_path}")

        lines.remove(matches[0])

        with open(serial_python_path, "w") as f:
            f.write("\n".join(lines))

        print(f"✅ Created: {serial_python_path}")
