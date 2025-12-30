#!/usr/bin/env python3
import argparse
import json
import pickle
from pathlib import Path
import os
import torch
import numpy as np
from torchvision import transforms
from tqdm import tqdm
from typing import List
from torch.utils.data import DataLoader

from src.models.get_model import get_pretrained_model
from src.data.read_data import read_input
from src.data.ImagePairDataset import ImagePairDataset
from src.evaluation.plot_utils import plot_age_distribution_heatmap, plot_prediction_error_heatmap
from src.data.ImagePairDataset import collate_fn


from src.constants import PRODUCTION_PLOTS, HYDRA_OUTPUT, PRODUCTION_CONFIGS_PATH


WEIGHTS = {
    "baseline":  Path("weights") / "age_resnet50.pth",
    "resnet50_1": HYDRA_OUTPUT / "resnet50_baseline" / "2025-12-30_12-56",
}


NAME_TO_CONFIG_PATH = {
    "baseline": Path(PRODUCTION_CONFIGS_PATH, "resnet50_baseline.pkl"),
}


parser = argparse.ArgumentParser(description="Process input and output file paths.")
parser.add_argument("input_file_path", type=str, help="Path to the input JSON file")
parser.add_argument("output_file_path", nargs="?", default=None, type=str, help="Path to the output file")
parser.add_argument("--brute", action="store_true", help="Evaluation in Brute")
parser.add_argument("--batch_size", type=int, default=32, help="Number of image pairs to process in a batch")
parser.add_argument("--num_workers", type=int, default=8, help="Number of worker processes for data loading")


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


def run_baseline_solution(
    data: dict,
    model,
    device: torch.device,
    args: argparse.Namespace,
) -> List:
    """
    Predicts the age of the person in the second image by using a prediction 
    error offset from the first image. It attempts to correct for person specific
    systematic under/over-estimation of the age.

    To execute this on a large dataset, the function uses a PyTorch `DataLoader`
    to efficiently process the images in batches on a target device (like a GPU),
    which significantly speeds up the computation.
    """
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = ImagePairDataset(data, preprocess)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,  # Do not shuffle for inference
        num_workers=args.num_workers,
        pin_memory=True,  # Helps speed up CPU to GPU data transfer
        collate_fn=collate_fn # Use our custom function to handle errors
    )

    results = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing Image Batches"):
            tensors1, tensors2, true_age1s, true_age1s = batch
            
            # Move data to the target device (GPU)
            tensors1 = tensors1.to(device)
            tensors2 = tensors2.to(device)
            true_age1s = true_age1s.to(device)
            
            final_predictions_age2 = model(tensors1, tensors2, true_age1s)
            results.extend([result.item() for result in final_predictions_age2])

    return results


def main(args):
    """
    Main function to run the age estimation pipeline.
    """
    data = read_input(args.input_file_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_name = "baseline"
    path_to_weights = WEIGHTS[model_name]
    path_to_cfg = NAME_TO_CONFIG_PATH[model_name]

    if args.brute:
        upload_dir = os.environ.get("UPLOAD_DIR", ".")
        path_to_weights = os.path.join(upload_dir, path_to_weights)
        path_to_cfg = os.path.join(upload_dir, path_to_cfg)

    with open(path_to_cfg, 'rb') as f:
        cfg = pickle.load(f)

    model = get_pretrained_model(cfg, path_to_weights, device)
    model.eval()

    predictions = run_baseline_solution(data, model, device, args)

    print(f"\nGenerated {len(predictions)} predictions.")
    
    if args.brute:
        save_list_to_path(predictions, args.output_file_path)
    else:
        print("Evaluating ...")
        try:
            with open(args.input_file_path.replace('instances', 'solutions'), 'r') as f:
                solution_data = json.load(f)

            true_ages1 = data['age1']
            ground_truth_ages2 = [p['face_2']['age'] for v in solution_data.values() for p in v]

            absolute_errors = [abs(pred - true) for pred, true in zip(predictions, ground_truth_ages2)]
            mae = np.mean(absolute_errors)
            print(f"\nMean Absolute Error (MAE): {mae:.4f}")
            
            # --- VISUALIZATIONS ---            
            plot_age_distribution_heatmap(true_ages1, ground_truth_ages2, PRODUCTION_PLOTS / model_name)
            plot_prediction_error_heatmap(true_ages1, ground_truth_ages2, predictions, PRODUCTION_PLOTS / model_name)

        except FileNotFoundError:
            print("\nCould not find the solution file to calculate MAE and generate plots.")
        except Exception as e:
            print(f"\nAn error occurred during evaluation: {e}")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)