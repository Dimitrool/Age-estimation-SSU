#!/bin/bash

# Define the name of the output zip file
OUTPUT_ZIP="project_archive.zip"

# Remove the old zip if it exists so we don't append to it
rm -f "$OUTPUT_ZIP"

echo "Creating archive..."

# Create the zip archive
zip -r -9 "$OUTPUT_ZIP" \
    main.py \
    src \
    pickle_configs \
    weights/age_resnet50.pth \
    -x "src/**/__pycache__/*" \
    -x "src/__pycache__/*" \
    -x "**/__pycache__/*" \
    -x "*.pyc" \
    -x "*.DS_Store" \
    -x "*.log" \
    -x ".git/*" \
    -x ".venv/*"

echo "-----------------------------------"
echo "Archive created successfully: $OUTPUT_ZIP"

# Output the size of the archive
du -h "$OUTPUT_ZIP"