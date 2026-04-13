import argparse
import os
import re

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", default="kernel", help="Directory to store generated pycompss files (default: kernel)")
args = parser.parse_args()

# Root directory containing your 12 folders
root_dir = args.output_dir

for subdir, _, files in os.walk(root_dir):
    if "omp" in files:
        omp_path = os.path.join(subdir, "omp")
        pycompss_path = os.path.join(subdir, "pycompss")

        with open(omp_path, "r") as f:
            content = f.read()

        # Find all comment blocks: /* ... */
        comments = re.findall(r"/\*(.*?)\*/", content, re.DOTALL)
        if not comments:
            print(f"No comment found in {omp_path}")
            continue

        # Take the LAST comment block (the most relevant one)
        comment_text = comments[-1].strip()

        # Replace 'OpenMP' with 'PyCOMPSs'
        comment_text = comment_text.replace("OpenMP", "PyCOMPSs")

        # Convert to Python-style triple-quoted comment
        py_comment = f'"""{comment_text}\n"""'

        with open(pycompss_path, "w") as f:
            f.write(py_comment)

        print(f"✅ Created: {pycompss_path}")
