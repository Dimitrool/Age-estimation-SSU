#!/usr/bin/env python3
import argparse
import json
import os
import torch
import numpy as np
from torchvision import transforms
from tqdm import tqdm
from typing import List, Dict, Optional
from torch.utils.data import DataLoader
from utils import save_list_to_path, read_input, ImagePairDataset, collate_fn
from plot_utils import plot_age_distribution_heatmap, plot_prediction_error_heatmap

parser = argparse.ArgumentParser(description="Process input and output file paths.")
parser.add_argument("input_file_path", type=str, help="Path to the input JSON file")
parser.add_argument("output_file_path", nargs="?", default=None, type=str, help="Path to the output file")
parser.add_argument("--brute", action="store_true", help="Evaluation in Brute")
parser.add_argument("--batch_size", type=int, default=32, help="Number of image pairs to process in a batch")
parser.add_argument("--num_workers", type=int, default=8, help="Number of worker processes for data loading")


def run_baseline_solution(
    data: dict,
    model: torch.jit.ScriptModule,
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
            tensors1, tensors2, true_age1s = batch
            
            # Move data to the target device (GPU)
            tensors1 = tensors1.to(device)
            tensors2 = tensors2.to(device)
            true_age1s = true_age1s.to(device)
            
            # Combine tensors for a single model pass: [face1_img1, ..., face1_imgN, face2_img1, ..., face2_imgN]
            input_batch = torch.cat([tensors1, tensors2])

            # Get model predictions
            risks, labels, posteriors = model.get_prediction(input_batch)
            predicted_ages = labels.get("age").squeeze()

            # Split predictions back into face1 and face2 groups
            num_valid_in_batch = tensors1.shape[0]
            predicted_age1s = predicted_ages[:num_valid_in_batch]
            predicted_age2s = predicted_ages[num_valid_in_batch:]

            # 1. Calculate offsets for the entire batch
            offsets = true_age1s - predicted_age1s

            # 2. Apply offsets to get final predictions. We only use half of the offset, a more conservative compensation
            final_predictions_age2 = predicted_age2s + offsets / 2.0
            
            # Place the results back into the correct positions in the main results list
            results.extend([result.item() for result in final_predictions_age2])
                
    return results


def main(args):
    """
    Main function to run the age estimation pipeline.
    """
    data = read_input(args.input_file_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_name = "age_resnet50.jit"
    if args.brute:
        upload_dir = os.environ.get("UPLOAD_DIR", ".")
        model_file_path = os.path.join(upload_dir, model_name)
    else:
        model_file_path = model_name

    jit_model = torch.jit.load(model_file_path, map_location=device)
    jit_model.eval()

    predictions = run_baseline_solution(data, jit_model, device, args)

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
            plot_age_distribution_heatmap(true_ages1, ground_truth_ages2)
            plot_prediction_error_heatmap(true_ages1, ground_truth_ages2, predictions)

        except FileNotFoundError:
            print("\nCould not find the solution file to calculate MAE and generate plots.")
        except Exception as e:
            print(f"\nAn error occurred during evaluation: {e}")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)