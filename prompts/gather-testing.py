import os
import json

def generate_json(root_dir, output_file):
    """
    Scans the directory structure and generates a JSON file.
    Structure: root_dir/[problem_type]/[name]/[filename]
    """
    data = []

    # Check if root directory exists
    if not os.path.exists(root_dir):
        print(f"Error: Directory '{root_dir}' not found.")
        return

    # Walk through problem types
    for problem_type in os.listdir(root_dir):
        pt_path = os.path.join(root_dir, problem_type)
        if not os.path.isdir(pt_path):
            continue
            
        # Walk through problem names
        for name in os.listdir(pt_path):
            n_path = os.path.join(pt_path, name)
            if not os.path.isdir(n_path):
                continue
                
            # Read each variation file inside the name folder
            for filename in os.listdir(n_path):
                file_path = os.path.join(n_path, filename)
                if os.path.isfile(file_path):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            prompt_content = f.read()
                        
                        # Construct the JSON entry
                        entry = {
                            "problem_type": problem_type,
                            "language": "python",
                            "name": name,
                            "parallelism_model": "pycompss",
                            "prompt": prompt_content
                        }
                        data.append(entry)
                    except Exception as e:
                        print(f"Could not read file {file_path}: {e}")

    # Write the list of objects to a JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    
    print(f"Successfully generated {output_file} with {len(data)} entries.")

if __name__ == "__main__":
    # Settings
    ROOT_FOLDER = 'raw_testing'
    OUTPUT_JSON = 'output.json'
    
    generate_json(ROOT_FOLDER, OUTPUT_JSON)