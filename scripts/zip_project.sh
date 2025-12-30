#!/bin/bash

# Define the name of the output zip file
OUTPUT_ZIP="project_archive.zip"

echo "Creating archive..."

# Create the zip archive
zip -r -9 "$OUTPUT_ZIP" \
    final \
    # main.py \
    # configs \
    # src \
    # -x "src/**/__pycache__/*" \
    # -x "src/__pycache__/*" \
    # -x "*.pyc" \
    # -x "*.DS_Store" \
    # -x "*.log"

echo "-----------------------------------"
echo "Archive created successfully: $OUTPUT_ZIP"

# Output the size of the archive
# du -h prints the size in human-readable format (e.g., 2.1M)
du -h "$OUTPUT_ZIP"
