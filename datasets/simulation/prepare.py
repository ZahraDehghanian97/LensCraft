import json
import glob

def merge_json_files(file_paths, output_file):
    merged_data = []

    for file_path in file_paths:
        with open(file_path, 'r') as file:
            data = json.load(file)
            merged_data.extend(data)
    
    with open(output_file, 'w') as output:
        json.dump(merged_data, output, indent=4)

input_files = glob.glob("./to-merge/*.json")
output_file = "merged_dataset.json"

merge_json_files(input_files, output_file)

print(f"Merged {len(input_files)} files into {output_file}")
