from typing import List


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

