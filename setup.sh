#!/bin/bash
# ============================================================
# CheXReport AI — Environment Setup Script
# Muhammed Panchla | Flowgenix AI
# ============================================================
# Run this script ONCE to set up your full environment.
# Usage: bash setup.sh

echo "============================================"
echo "  CheXReport AI — Setup Script"
echo "  Muhammed Panchla | Flowgenix AI"
echo "============================================"

# Step 1: Create virtual environment
echo ""
echo "Step 1: Creating virtual environment..."
python3 -m venv chexenv
source chexenv/bin/activate
echo "✅ Virtual environment created: chexenv"

# Step 2: Install dependencies
echo ""
echo "Step 2: Installing dependencies..."
pip install --upgrade pip -q
pip install torch torchvision torchaudio -q
pip install transformers -q
pip install jupyter notebook -q
pip install pandas numpy matplotlib -q
pip install Pillow scikit-learn -q
pip install nltk -q
pip install gradio -q
echo "✅ All dependencies installed"

# Step 3: Create folder structure
echo ""
echo "Step 3: Creating project folders..."
mkdir -p data/images data/reports models notebooks src weights
echo "✅ Folders created"

# Step 4: Download IU X-Ray dataset
echo ""
echo "Step 4: Downloading IU X-Ray Dataset..."
echo "Source: NIH Open-i (openi.nlm.nih.gov)"
echo ""

# Download images
echo "Downloading chest X-ray images (~1.2GB)..."
curl -L "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz" -o NLMCXR_png.tgz
echo "Extracting images..."
tar -xzf NLMCXR_png.tgz -C data/images/
rm NLMCXR_png.tgz
echo "✅ Images downloaded and extracted"

# Download reports
echo ""
echo "Downloading radiology reports..."
curl -L "https://openi.nlm.nih.gov/imgs/collections/ecgen-radiology.tar.gz" -o ecgen-radiology.tar.gz
echo "Extracting reports..."
tar -xzf ecgen-radiology.tar.gz -C data/reports/
rm ecgen-radiology.tar.gz
echo "✅ Reports downloaded and extracted"

# Step 5: Verify
echo ""
echo "Step 5: Verifying download..."
IMAGE_COUNT=$(find data/images -name "*.png" | wc -l)
REPORT_COUNT=$(find data/reports -name "*.xml" | wc -l)
echo "Images found: $IMAGE_COUNT"
echo "Reports found: $REPORT_COUNT"

if [ "$IMAGE_COUNT" -gt 7000 ]; then
    echo "✅ Images OK"
else
    echo "⚠️  Warning: Expected ~7470 images, got $IMAGE_COUNT"
fi

if [ "$REPORT_COUNT" -gt 3000 ]; then
    echo "✅ Reports OK"
else
    echo "⚠️  Warning: Expected ~3955 reports, got $REPORT_COUNT"
fi

# Step 6: Launch Jupyter
echo ""
echo "============================================"
echo "✅ Setup Complete!"
echo ""
echo "To start working:"
echo "  source chexenv/bin/activate"
echo "  cd notebooks"
echo "  jupyter notebook"
echo ""
echo "Open: 01_data_exploration.ipynb first"
echo "============================================"
