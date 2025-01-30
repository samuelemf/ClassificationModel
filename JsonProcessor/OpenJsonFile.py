import json


def openJsonFile(json_file_path):
    with open(json_file_path, 'r') as f:
        return json.load(f)
