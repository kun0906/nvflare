import os
import sys
import json


def format_all_json_files(directory):
    """Find and format all JSON files in a directory."""
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('cross_val_results.json'):
                file_path = os.path.join(root, file)
                try:
                    format_json(file_path)
                except Exception as e:
                    print(f"Error formatting {file_path}: {e}")


def extract_number(key):
    """Extract numeric part from the dictionary key."""
    try:
        res = int(key.split('-')[1])
    except Exception as e:
        # print(e, key)
        res = -1
    return res


def sort_dict_numerically(data):
    """Sort dictionary keys numerically."""
    if isinstance(data, dict):
        sorted_dict = dict(sorted(data.items(), key=lambda item: extract_number(item[0])))
        return {k: sort_dict_numerically(v) for k, v in sorted_dict.items()}
    elif isinstance(data, list):
        return [sort_dict_numerically(i) for i in data]
    else:
        return data


def format_json(json_file):
    print(f'\n\n, formatting {json_file}')
    # Load JSON from a file
    with open(json_file, 'r') as file:
        data = json.load(file)

    # Format the JSON with indentation for pretty printing
    formatted_json = json.dumps(data, indent=4, separators=(',', ': '), sort_keys=True)

    # # Sort dictionary keys numerically if the data is a dictionary
    # data = sort_dict_numerically(data)
    # formatted_json = json.dumps(data, indent=4, separators=(',', ': '), sort_keys=False)
    # Print the formatted JSON
    print(formatted_json)
    # Write the formatted JSON back to the same file
    with open(json_file, 'w') as file:
        file.write(formatted_json)


if __name__ == '__main__':
    # json_file = sys.argv[1]
    # json_file = '/Users/49751124/PycharmProjects/nvflare/def38e5d-b037-41e2-9566-ec7c7f3074b7/workspace/cross_site_val/cross_val_results.json'
    # format_json(json_file)

    in_dir = os.path.expanduser('~/cifar10-hello-pt-10clients-2classes/transfer')
    format_all_json_files(in_dir)
