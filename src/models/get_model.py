import torch






def get_model(model_file_path, device):
    # both the model's structure (code) and its parameters (weights) are stored
    if ".jit" in model_file_path:
        return torch.jit.load(model_file_path, map_location=device)
    
    

