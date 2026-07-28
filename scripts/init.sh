#!/usr/bin/env bash
# -------------------------------
# Zenodo dataset
# -------------------------------
RECORD_ID=21649988

# -------------------------------
# Download directory
# -------------------------------
SCRIPT_DIR="$(pwd)"

echo "Working directory: ${SCRIPT_DIR}"

# -------------------------------
# Download files
# -------------------------------
echo "Downloading data.zip..."
wget -c "https://zenodo.org/records/${RECORD_ID}/files/data.zip"

echo "Downloading opt_data.zip..."
wget -c "https://zenodo.org/records/${RECORD_ID}/files/opt_data.zip"

echo "Downloading synthesis_data.zip..."
wget -c "https://zenodo.org/records/${RECORD_ID}/files/synthesis_data.zip"

# -------------------------------
# Create extraction folders
# -------------------------------
echo "Creating directories..."

# -------------------------------
# Extract files
# -------------------------------
cd ${SCRIPT_DIR}/cycle_count_estimation
#
unzip "../data.zip"
#
cd ${SCRIPT_DIR}
unzip opt_data.zip
#
cd ${SCRIPT_DIR}/synthesis_estimation
unzip "../synthesis_data.zip"

echo ""
echo "================================"
echo "Dataset extraction complete"
echo "================================"