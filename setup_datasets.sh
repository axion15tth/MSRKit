#!/bin/bash
# Setup script for MSRKit datasets
# Usage: ./setup_datasets.sh /path/to/datasets/parent/dir

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== MSRKit Dataset Setup ===${NC}"
echo

# Check argument
if [ $# -eq 0 ]; then
    echo -e "${YELLOW}Usage: $0 /path/to/datasets/parent/dir${NC}"
    echo
    echo "Example: If you have:"
    echo "  /data/MUSDB18-HQ/"
    echo "  /data/MoisesDB/"
    echo "  /data/RawStems-48k/"
    echo
    echo "Run: $0 /data"
    echo
    exit 1
fi

DATASETS_PARENT="$1"

if [ ! -d "$DATASETS_PARENT" ]; then
    echo -e "${RED}Error: Directory $DATASETS_PARENT does not exist${NC}"
    exit 1
fi

echo "Datasets parent directory: $DATASETS_PARENT"
echo

# Create data directory if not exists
mkdir -p data

# Function to create symlink
create_link() {
    local source="$1"
    local target="$2"
    local name="$3"
    
    if [ -d "$source" ] || [ -L "$source" ]; then
        echo -e "${GREEN}✓${NC} Found $name at: $source"
        if [ ! -L "$target" ] && [ ! -d "$target" ]; then
            ln -s "$source" "$target"
            echo "  → Created symlink: $target"
        elif [ -L "$target" ]; then
            echo "  → Symlink already exists: $target"
        else
            echo "  → Directory already exists: $target"
        fi
        return 0
    else
        echo -e "${YELLOW}⚠${NC} $name not found at: $source"
        return 1
    fi
}

# Detect and link datasets
FOUND_DATASETS=()

echo "Searching for datasets..."
echo

# MUSDB18 variants
for variant in MUSDB18-HQ MUSDB18-48k MUSDB18; do
    if create_link "$DATASETS_PARENT/$variant" "data/$variant" "$variant"; then
        FOUND_DATASETS+=("data/$variant")
        break
    fi
done

# MoisesDB variants
for variant in MoisesDB-48k MoisesDB moisesdb; do
    if create_link "$DATASETS_PARENT/$variant" "data/MoisesDB-48k" "MoisesDB"; then
        FOUND_DATASETS+=("data/MoisesDB-48k")
        break
    fi
done

# RawStems variants
for variant in RawStems-48k RawStems rawstems; do
    if create_link "$DATASETS_PARENT/$variant" "data/RawStems-48k" "RawStems"; then
        FOUND_DATASETS+=("data/RawStems-48k")
        break
    fi
done

echo
echo "================================"

if [ ${#FOUND_DATASETS[@]} -eq 0 ]; then
    echo -e "${RED}Error: No datasets found in $DATASETS_PARENT${NC}"
    echo
    echo "Expected directory structure:"
    echo "  $DATASETS_PARENT/"
    echo "    ├── MUSDB18-HQ/ (or MUSDB18-48k/)"
    echo "    ├── MoisesDB-48k/ (or MoisesDB/)"
    echo "    └── RawStems-48k/ (or RawStems/)"
    exit 1
fi

echo -e "${GREEN}Found ${#FOUND_DATASETS[@]} dataset(s)${NC}"
echo

# Generate filelists
echo "Generating train/val file lists..."

# Check if python is available
if ! command -v python &> /dev/null; then
    echo -e "${RED}Error: python command not found${NC}"
    echo "Please activate your virtual environment:"
    echo "  source venv/bin/activate"
    exit 1
fi

# Build filelists
python tools/build_filelists.py \
    --roots "${FOUND_DATASETS[@]}" \
    --out_dir lists \
    --val_ratio 0.1 \
    --seed 1337

echo
if [ -f "lists/train_vocals.txt" ] && [ -f "lists/val_vocals.txt" ]; then
    TRAIN_COUNT=$(wc -l < lists/train_vocals.txt)
    VAL_COUNT=$(wc -l < lists/val_vocals.txt)
    echo -e "${GREEN}✓ File lists generated successfully${NC}"
    echo "  Train: $TRAIN_COUNT samples"
    echo "  Val:   $VAL_COUNT samples"
else
    echo -e "${RED}✗ Failed to generate file lists${NC}"
    exit 1
fi

echo
echo "================================"
echo -e "${GREEN}Setup complete!${NC}"
echo
echo "Next steps:"
echo "  1. Review config.yaml (data paths should be auto-detected)"
echo "  2. Start training:"
echo "     python train.py --config config.yaml"
echo
