#!/bin/bash
# Setup script for MSRKit datasets
# Supports individual dataset paths or a parent directory

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== MSRKit Dataset Setup ===${NC}"
echo

# Usage function
show_usage() {
    echo -e "${YELLOW}Usage:${NC}"
    echo "  Option 1 (Individual paths):"
    echo "    $0 --musdb /path/to/MUSDB18-HQ --moisesdb /path/to/MoisesDB --rawstems /path/to/RawStems"
    echo
    echo "  Option 2 (Parent directory):"
    echo "    $0 /parent/directory"
    echo
    echo "  Option 3 (Environment variables):"
    echo "    export MUSDB_PATH=/path/to/MUSDB18-HQ"
    echo "    export MOISESDB_PATH=/path/to/MoisesDB"
    echo "    export RAWSTEMS_PATH=/path/to/RawStems"
    echo "    $0"
    echo
    echo -e "${YELLOW}Options:${NC}"
    echo "  --musdb PATH       Path to MUSDB18-HQ dataset"
    echo "  --moisesdb PATH    Path to MoisesDB dataset"
    echo "  --rawstems PATH    Path to RawStems dataset"
    echo "  --help             Show this help message"
    echo
    echo -e "${YELLOW}Examples:${NC}"
    echo "  # All datasets in one parent directory"
    echo "  $0 /data"
    echo
    echo "  # Datasets in different locations"
    echo "  $0 --musdb /datasets/musdb --moisesdb /mnt/moises --rawstems /data/rawstems"
    echo
}

# Parse arguments
MUSDB_PATH=""
MOISESDB_PATH=""
RAWSTEMS_PATH=""
PARENT_DIR=""

if [ $# -eq 0 ]; then
    # Try environment variables
    MUSDB_PATH="${MUSDB_PATH:-}"
    MOISESDB_PATH="${MOISESDB_PATH:-}"
    RAWSTEMS_PATH="${RAWSTEMS_PATH:-}"

    if [ -z "$MUSDB_PATH" ] && [ -z "$MOISESDB_PATH" ] && [ -z "$RAWSTEMS_PATH" ]; then
        show_usage
        exit 1
    fi
elif [ $# -eq 1 ]; then
    if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
        show_usage
        exit 0
    fi
    PARENT_DIR="$1"
else
    while [[ $# -gt 0 ]]; do
        case $1 in
            --musdb)
                MUSDB_PATH="$2"
                shift 2
                ;;
            --moisesdb)
                MOISESDB_PATH="$2"
                shift 2
                ;;
            --rawstems)
                RAWSTEMS_PATH="$2"
                shift 2
                ;;
            --help|-h)
                show_usage
                exit 0
                ;;
            *)
                echo -e "${RED}Error: Unknown option $1${NC}"
                show_usage
                exit 1
                ;;
        esac
    done
fi

# Create data directory if not exists
mkdir -p data

# Function to find dataset
find_dataset() {
    local base_path="$1"
    local variants=("${@:2}")

    for variant in "${variants[@]}"; do
        if [ -d "$base_path/$variant" ]; then
            echo "$base_path/$variant"
            return 0
        fi
    done

    # Check if base_path itself matches
    for variant in "${variants[@]}"; do
        if [[ "$base_path" == *"$variant"* ]] && [ -d "$base_path" ]; then
            echo "$base_path"
            return 0
        fi
    done

    return 1
}

# Function to create symlink
create_link() {
    local source="$1"
    local target="$2"
    local name="$3"

    if [ -z "$source" ] || [ ! -d "$source" ]; then
        echo -e "${YELLOW}⚠${NC} $name not found or not specified"
        return 1
    fi

    # Convert to absolute path
    source=$(cd "$source" && pwd)

    echo -e "${GREEN}✓${NC} Found $name at: $source"

    if [ -L "$target" ]; then
        # Symlink exists - check if it points to the same location
        existing=$(readlink -f "$target")
        if [ "$existing" == "$source" ]; then
            echo "  → Symlink already exists: $target"
        else
            echo "  → Updating symlink: $target"
            rm "$target"
            ln -s "$source" "$target"
        fi
    elif [ -d "$target" ] && [ ! -L "$target" ]; then
        echo "  → Directory already exists: $target"
    else
        ln -s "$source" "$target"
        echo "  → Created symlink: $target"
    fi
    return 0
}

# Detect datasets
FOUND_DATASETS=()

echo -e "${BLUE}Searching for datasets...${NC}"
echo

# If parent directory is specified, search for datasets
if [ -n "$PARENT_DIR" ]; then
    if [ ! -d "$PARENT_DIR" ]; then
        echo -e "${RED}Error: Directory $PARENT_DIR does not exist${NC}"
        exit 1
    fi

    echo "Parent directory: $PARENT_DIR"
    echo

    # Auto-detect datasets in parent directory
    if [ -z "$MUSDB_PATH" ]; then
        MUSDB_PATH=$(find_dataset "$PARENT_DIR" "MUSDB18-HQ" "MUSDB18-48k" "MUSDB18" "musdb18-hq" "musdb18") || true
    fi
    if [ -z "$MOISESDB_PATH" ]; then
        MOISESDB_PATH=$(find_dataset "$PARENT_DIR" "MoisesDB-48k" "MoisesDB" "moisesdb-48k" "moisesdb") || true
    fi
    if [ -z "$RAWSTEMS_PATH" ]; then
        RAWSTEMS_PATH=$(find_dataset "$PARENT_DIR" "RawStems-48k" "RawStems" "rawstems-48k" "rawstems") || true
    fi
fi

# Create symlinks
if create_link "$MUSDB_PATH" "data/MUSDB18-HQ" "MUSDB18-HQ"; then
    FOUND_DATASETS+=("data/MUSDB18-HQ")
fi

if create_link "$MOISESDB_PATH" "data/MoisesDB-48k" "MoisesDB"; then
    FOUND_DATASETS+=("data/MoisesDB-48k")
fi

if create_link "$RAWSTEMS_PATH" "data/RawStems-48k" "RawStems"; then
    FOUND_DATASETS+=("data/RawStems-48k")
fi

echo
echo "================================"

if [ ${#FOUND_DATASETS[@]} -eq 0 ]; then
    echo -e "${RED}Error: No datasets found${NC}"
    echo
    echo "Please specify dataset paths using one of these methods:"
    echo "  1. Individual paths: --musdb, --moisesdb, --rawstems"
    echo "  2. Parent directory: $0 /path/to/parent"
    echo "  3. Environment variables: MUSDB_PATH, MOISESDB_PATH, RAWSTEMS_PATH"
    echo
    show_usage
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
echo "  1. Review config.yaml if needed"
echo "  2. Start training:"
echo "     python train.py --config config.yaml"
echo
