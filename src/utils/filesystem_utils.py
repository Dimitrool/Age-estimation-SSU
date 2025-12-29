import json
from typing import Dict, List


def save_list_to_path(list: List, output_file_path: str) -> None:
    """
    Save a list of numbers to a text file.
    
    Args:
        list (List): The list to save.
        output_file_path (str): Path to the output file.
    """
    # Open the output file to write the results
    with open(output_file_path, 'w') as outfile:
        # Write the values separated by spaces
        outfile.write(" ".join([str(val) for val in list]) + "\n")
    

def read_input(input_file_path: str) -> Dict[str, List]:
    """
    Read input data.
    
    Args:
        input_file_path (str): Path to the input file.
    
    Returns:
        dict: A dictionary with keys "face1_path", "age1", and "face2_path",
              each containing a flat list of data.
    """
    with open(input_file_path, "r") as f:
        nested_data = json.load(f)

    face1_paths = []
    true_age1s = []
    face2_paths = []
    true_age2s = []

    # Iterate through each creator and their list of pairs
    for creator_id, pairs in nested_data.items():
        for pair in pairs:
            person_1 = pair['face_1']
            person_2 = pair['face_2']

            face1_paths.append(person_1['image_path'])
            true_age1s.append(person_1['age'])
            face2_paths.append(person_2['image_path'])
            true_age2s.append(person_2['age'])

    flat_data = {
        "face1_path": face1_paths,
        "age1": true_age1s,
        "face2_path": face2_paths,
        "age2": true_age2s
    }
    
    return flat_data
