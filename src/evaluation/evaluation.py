import json
import numpy as np

from src.evaluation.plot_utils import plot_age_distribution_heatmap, plot_prediction_error_heatmap


def evaluate(predictions, input_file_path, result_path):
    print(f"\nGenerated {len(predictions)} predictions.")
    print("Evaluating ...")

    try:
        with open(input_file_path.replace('instances', 'solutions'), 'r') as f:
            solution_data = json.load(f)

        true_ages1 = solution_data['age1']
        ground_truth_ages2 = [p['face_2']['age'] for v in solution_data.values() for p in v]

        absolute_errors = [abs(pred - true) for pred, true in zip(predictions, ground_truth_ages2)]
        mae = np.mean(absolute_errors)
        print(f"\nMean Absolute Error (MAE): {mae:.4f}")
        
        # --- VISUALIZATIONS ---            
        plot_age_distribution_heatmap(true_ages1, ground_truth_ages2, result_path)
        plot_prediction_error_heatmap(true_ages1, ground_truth_ages2, predictions, result_path)

    except FileNotFoundError:
        print("\nCould not find the solution file to calculate MAE and generate plots.")
    except Exception as e:
        print(f"\nAn error occurred during evaluation: {e}")
