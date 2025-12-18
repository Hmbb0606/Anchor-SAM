#!/bin/bash

# ==========================================
# Script to automatically download ResNet pretrained weights
# ==========================================

# Base download URL
BASE_URL="https://download.pytorch.org/models"

# Define the list of filenames to download
# Note: resnet50-0676ba61 is an old hash, the new version is usually resnet50-19c8e357
# The script attempts to satisfy your specific filename requirement first
FILES=(
    "resnet18-f37072fd.pth"
    "resnet34-b627a593.pth"
    "resnet50-0676ba61.pth"
    "resnet101-5d3b4d8f.pth"
)

# ResNet50 fallback hash (If old 0676ba61 fails, try this and rename)
RESNET50_NEW="resnet50-19c8e357.pth"

echo "---------------------------------------"
echo "Starting check and download of ResNet pretrained weights..."
echo "Save path: $(pwd)"
echo "---------------------------------------"

# Check for download tool
if command -v wget &> /dev/null; then
    DOWNLOADER="wget"
    echo "Using wget for download."
elif command -v curl &> /dev/null; then
    DOWNLOADER="curl"
    echo "wget not found, using curl for download."
else
    echo "Error: Neither wget nor curl found. Cannot download."
    exit 1
fi

# Loop through files
for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ [Exists] $file - Skipping"
    else
        echo "⬇️  [Downloading] $file ..."

        # Build download URL
        URL="$BASE_URL/$file"

        # Execute download command
        if [ "$DOWNLOADER" == "wget" ]; then
            wget -c "$URL" -O "$file"
        else
            curl -C - -o "$file" "$URL"
        fi

        # Check if download was successful
        if [ $? -eq 0 ]; then
            echo "✅ $file download complete."
        else
            echo "⚠️  $file download failed (link might be outdated)."

            # Special handling for ResNet50
            if [[ "$file" == "resnet50-0676ba61.pth" ]]; then
                echo "🔄 Attempting to download ResNet50 new version ($RESNET50_NEW) and rename..."
                URL_NEW="$BASE_URL/$RESNET50_NEW"

                if [ "$DOWNLOADER" == "wget" ]; then
                    wget -c "$URL_NEW" -O "$file"
                else
                    curl -C - -o "$file" "$URL_NEW"
                fi

                if [ $? -eq 0 ]; then
                    echo "✅ Download and rename successful: $file (Content is new version)"
                else
                    echo "❌ ResNet50 download failed completely."
                    rm -f "$file" # Remove empty file
                fi
            else
                echo "❌ Download failed, please check network or filename."
                rm -f "$file" # Remove empty file
            fi
        fi
    fi
    echo "---------------------------------------"
done

echo "🎉 All operations completed."