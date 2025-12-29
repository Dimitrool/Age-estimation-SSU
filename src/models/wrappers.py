import torch
import torch.nn as nn


from src.constants import MAX_AGE


class BaseBackboneWrapper(nn.Module):
    """
    Base class that handles dynamic feature dimension calculation.
    """
    def __init__(self, backbone: nn.Module, img_size: int):
        super().__init__()
        # Child classes must define self.features and self.head
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.head = nn.Sequential(list(backbone.children())[-1])
        self.feature_dim = 0

        self._calculate_feature_dim(img_size)

        
    def _calculate_feature_dim(self, img_size: int):
        """
        Runs a dummy forward pass to determine the output dimension of the feature extractor.
        """

        dummy_input = torch.zeros(1, 3, img_size, img_size)
        with torch.no_grad():
            output = self.features(dummy_input)
            # Flatten to [Batch, Dim] to check the size
            output = torch.flatten(output, 1)
            self.feature_dim = output.shape[1]


class ResNet50BaselineWrapper(BaseBackboneWrapper):
    def __init__(self, backbone: nn.Module, img_size: int):
        super().__init__(backbone, img_size)
        self.age_indices = torch.arange(MAX_AGE).float()
    
    def forward(self, img, return_embeddings=False):
        # 1. Extract Embeddings (The Context)
        # [Batch, 2048, 1, 1]
        x = self.features(img)
        # Flatten to [Batch, 2048]
        embeddings = torch.flatten(x, 1)

        # 2. Get Logits (The Answer)
        logits = self.head(embeddings)
    
        # Get model predictions
        posteriors = torch.nn.functional.softmax(logits, dim=1)

        if self.training:
            # Use "Expected Value" (weighted average) so gradients can flow
            predicted_age = torch.sum(posteriors * self.age_indices, dim=1)
        else:
            # The median is the first class index where the CDF is >= 0.5
            # We use argmax on the boolean tensor, as it returns the index of the first 'True' value.
            cdf = torch.cumsum(posteriors, dim=1)
            predicted_age = torch.argmax((cdf >= 0.5).int(), dim=1).float()

        if return_embeddings:
            return predicted_age, embeddings

        return predicted_age


class ConstOffsetCorrectionWrapper(nn.Module):
    def __init__(self, backbone: nn.Module, factor: float = 0.5):
        super().__init__()
        self.backbone = backbone
        self.factor = factor

    def forward(self, img1, img2, true_age1):
        pred_age1 = self.backbone(img1)
        pred_age2 = self.backbone(img2)
        
        current_bias = (true_age1 - pred_age1).detach()
        
        return pred_age2 + (current_bias * self.factor)


class LearnableOffsetCorrectionWrapper(nn.Module):
    def __init__(self, base_model: nn.Module, error_mapper: nn.Module):
        super().__init__()
        self.base_model = base_model
        self.error_mapper = error_mapper
    
    def forward(self, img1, img2, true_age1):
        pred_age1 = self.base_model(img1)
        pred_age2 = self.base_model(img2)
        
        error1 = true_age1 - pred_age1
        
        predicted_correction = self.error_mapper(error1)

        return pred_age2 + predicted_correction


class DeltaRegressionWrapper(nn.Module):
    def __init__(self, backbone: nn.Module, delta_head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.delta_head = delta_head

    def forward(self, img1, img2, true_age1):
        # 1. Get features
        _, feat1 = self.backbone(img1, return_embeddings=True).view(img1.size(0), -1)
        _, feat2 = self.backbone(img2, return_embeddings=True).view(img2.size(0), -1)
        
        # 2. Calculate the "Aging Vector"
        # This removes identity info (static features) and leaves only changes
        feature_delta = feat2 - feat1
        
        # 3. Predict how many years correspond to this feature change
        age_delta = self.delta_head(feature_delta)
        
        # 4. Apply to the known age
        return true_age1 + age_delta




