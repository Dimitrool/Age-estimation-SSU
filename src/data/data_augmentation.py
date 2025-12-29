from torchvision import transforms


TRANSFORMS_REGISTRY = {
    # --- GEOMETRIC TRANSFORMATIONS ---
    # Resizes the image to the given size.
    "resize": transforms.Resize,
    
    # Crops the given image at the center.
    "center_crop": transforms.CenterCrop,
    
    # Crops the image at a random location.
    "random_crop": transforms.RandomCrop,
    
    # Crops a random portion of image and resizes it to a given size.
    # (Great for training Inception/ResNet models).
    "random_resized_crop": transforms.RandomResizedCrop,
    
    # Horizontally flips the given image randomly with a given probability.
    "random_horizontal_flip": transforms.RandomHorizontalFlip,
    
    # Vertically flips the given image randomly with a given probability.
    "random_vertical_flip": transforms.RandomVerticalFlip,
    
    # Rotates the image by angle.
    "random_rotation": transforms.RandomRotation,
    
    # Affine transformation of the image keeping center invariant.
    "random_affine": transforms.RandomAffine,
    
    # Pad the given image on all sides with the given "padding" value.
    "pad": transforms.Pad,
    
    # Performs a random perspective transformation of the given image.
    "random_perspective": transforms.RandomPerspective,


    # --- COLOR & APPEARANCE TRANSFORMATIONS ---
    # Randomly changes the brightness, contrast, saturation and hue.
    "color_jitter": transforms.ColorJitter,
    
    # Convert image to grayscale.
    "grayscale": transforms.Grayscale,
    
    # Randomly convert image to grayscale with a probability of p.
    "random_grayscale": transforms.RandomGrayscale,
    
    # Blurs image with randomly chosen Gaussian blur.
    "gaussian_blur": transforms.GaussianBlur,
    
    # Inverts the colors of the given image randomly with a given probability.
    "random_invert": transforms.RandomInvert,
    
    # Posterizes the image randomly with a given probability (reduces bits).
    "random_posterize": transforms.RandomPosterize,
    
    # Solarizes the image randomly with a given probability (inverts pixels above threshold).
    "random_solarize": transforms.RandomSolarize,
    
    # Adjust the sharpness of the image randomly with a given probability.
    "random_adjust_sharpness": transforms.RandomAdjustSharpness,
    
    # Autocontrast the pixels of the given image randomly with a given probability.
    "random_autocontrast": transforms.RandomAutocontrast,
    
    # Equalize the histogram of the given image randomly with a given probability.
    "random_equalize": transforms.RandomEqualize,


    # --- TENSOR & CONVERSION ---
    # Convert a PIL Image or numpy.ndarray to tensor.
    "to_tensor": transforms.ToTensor,
    
    # Normalize a tensor image with mean and standard deviation.
    "normalize": transforms.Normalize,
    
    # Randomly selects a rectangle region in an image tensor and erases its pixels.
    # NOTE: This must come AFTER ToTensor().
    "random_erasing": transforms.RandomErasing,
}


def build_transforms(augmentation_cfg: dict, img_size: int):
    """
    Builds a pipeline with mandatory defaults, injecting user augmentations
    in the correct position (Middle for PIL, End for Tensors).
    """
    pipeline: list = [
        transforms.Resize(img_size)
    ]

    post_normalization_transforms = []
    for item in augmentation_cfg:
        name = item['name']
        kwargs = item['params']
        
        # Safety: Skip defaults if they are accidentally added to config
        if name in ["resize", "to_tensor", "normalize"]:
            continue
        
        # Special Handling: RandomErasing must happen AFTER Normalize
        if name == "random_erasing":
            if name in TRANSFORMS_REGISTRY:
                post_normalization_transforms.append(TRANSFORMS_REGISTRY[name](**kwargs))
            continue

        # Standard augmentations (Flip, Rotate, Color) go here
        if name in TRANSFORMS_REGISTRY:
            pipeline.append(TRANSFORMS_REGISTRY[name](**kwargs))
        else:
            raise ValueError(f"Transform {name} not found in registry")
            
    # 3. Mandatory End: Convert to Math Object (PIL -> Tensor)
    pipeline.append(transforms.ToTensor())
    pipeline.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    # 4. Append Tensor-specific transforms (like RandomErasing)
    pipeline.extend(post_normalization_transforms)

    return transforms.Compose(pipeline)
