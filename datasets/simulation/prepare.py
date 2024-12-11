import json
import glob

def merge_json_files(file_paths, output_file):
    merged_data = {
        "simulations": [],
        "metadata": {
            "simulationCount": 0,
            "subjectCount": 0,
            "instructionCount": 0,
            "minFrameCount": 30,
            "maxFrameCount": 30
        }
    }

    for file_path in file_paths:
        with open(file_path, 'r') as file:
            data = json.load(file)
            
            merged_data["simulations"].extend(data.get("simulations", []))
            
            merged_data["metadata"]["simulationCount"] += data["metadata"].get("simulationCount", 0)
            merged_data["metadata"]["subjectCount"] += data["metadata"].get("subjectCount", 0)
            merged_data["metadata"]["instructionCount"] += data["metadata"].get("instructionCount", 0)
    
    with open(output_file, 'w') as output:
        json.dump(merged_data, output, indent=4)

input_files = glob.glob("./to-merge/*.json")
output_file = "merged_dataset.json"

merge_json_files(input_files, output_file)

print(f"Merged {len(input_files)} files into {output_file}")
