import json
from typing import Dict, List


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
    person_id = []

    # Iterate through each creator and their list of pairs
    for creator_id, pairs in nested_data.items():
        for pair in pairs:
            person_1 = pair['face_1']
            person_2 = pair['face_2']

            face1_paths.append(person_1['image_path'])
            true_age1s.append(person_1['age'])
            face2_paths.append(person_2['image_path'])
            true_age2s.append(person_2['age'])
            person_id.append(creator_id)

    flat_data = {
        "face1_path": face1_paths,
        "age1": true_age1s,
        "face2_path": face2_paths,
        "age2": true_age2s,
        "person_id": person_id,
    }
    
    return flat_data


def read_self_augmented_input(input_file_path: str) -> Dict[str, List]:
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
    person_id = []

    # Iterate through each creator and their list of pairs
    for creator_id, pairs in nested_data.items():
        pairs_num = len(pairs)

        for i in range(pairs_num):
            for j in range(pairs_num):
                person_1 = pairs[i]['face_1']
                person_2 = pairs[j]['face_2']

                face1_paths.append(person_1['image_path'])
                true_age1s.append(person_1['age'])
                face2_paths.append(person_2['image_path'])
                true_age2s.append(person_2['age'])
                person_id.append(creator_id)

    flat_data = {
        "face1_path": face1_paths,
        "age1": true_age1s,
        "face2_path": face2_paths,
        "age2": true_age2s,
        "person_id": person_id,
    }
    
    return flat_data
