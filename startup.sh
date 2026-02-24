#!/bin/bash
# Pod Startup Script for SD WebUI Forge Classic (Neo) + Flux.1 Dev

set -e

WORKSPACE="/workspace"
REPO_DIR="$WORKSPACE/sd-webui-forge-classic"
REPO_URL="https://github.com/nolanlane/sd-webui-forge-classic.git"
BRANCH="main"
MODEL_DIR="$REPO_DIR/models/Stable-diffusion"
MODEL_URL="https://huggingface.co/lllyasviel/flux1_dev/resolve/main/flux1-dev-fp8.safetensors"
MODEL_FILE="flux1-dev-fp8.safetensors"

echo "=========================================="
echo "Starting Forge Pre-flight checks..."
echo "=========================================="

# 1. OS Dependencies
MISSING_PKGS=""
for pkg in curl git libgl1 libglib2.0-0 ffmpeg; do
    if ! dpkg -s $pkg >/dev/null 2>&1; then
        MISSING_PKGS="$MISSING_PKGS $pkg"
    fi
done

if [ -n "$MISSING_PKGS" ]; then
    echo "Installing missing OS dependencies:$MISSING_PKGS"
    apt-get update && apt-get install -y $MISSING_PKGS
else
    echo "OS dependencies are already installed."
fi

# 2. uv Package Manager
if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
else
    echo "uv package manager is already installed."
fi

# 3. Clone Repository
if [ ! -d "$REPO_DIR" ]; then
    echo "Cloning Forge Classic (Neo branch) into $WORKSPACE..."
    git clone --branch $BRANCH $REPO_URL $REPO_DIR
else
    echo "Repository already exists at $REPO_DIR. Checking for updates..."
    # Optionally update the repo: cd $REPO_DIR && git pull origin $BRANCH
fi

# 4. Setup Python 3.13 & Virtual Environment
echo "Checking Python 3.13 installation..."
if ! uv python list | grep -q '3.13'; then
    echo "Installing Python 3.13 via uv..."
    uv python install 3.13
fi

cd "$REPO_DIR"

# Nuke the venv if it exists but has the wrong Python version (e.g., 3.10)
if [ -d ".venv" ]; then
    VENV_PYTHON_VERSION=$(.venv/bin/python --version 2>&1 || echo "None")
    if [[ "$VENV_PYTHON_VERSION" != *"3.13"* ]]; then
        echo "🚨 Found existing .venv with INCORRECT Python version ($VENV_PYTHON_VERSION). Deleting to recreate with 3.13..."
        rm -rf .venv
    fi
fi

if [ ! -d ".venv" ]; then
    echo "Creating persistent virtual environment with Python 3.13..."
    uv venv .venv --python 3.13 --seed
else
    echo "✅ Valid virtual environment with Python 3.13 already exists."
fi

# 5. Download Flux.1 Dev Model & Dependencies
download_file() {
    local url="$1"
    local dest="$2"
    local tmp="${dest}.part"

    echo "Downloading: $url"
    echo " -> $dest"

    rm -f "$tmp"
    if ! curl -L --fail --retry 3 --retry-delay 2 --connect-timeout 20 --max-time 0 \
        --progress-bar -o "$tmp" "$url"; then
        echo "❌ Download failed: $url"
        rm -f "$tmp"
        return 1
    fi

    local size
    size=$(stat -c%s "$tmp" 2>/dev/null || echo 0)
    if [ "$size" -le 0 ]; then
        echo "❌ Download produced empty file: $dest"
        rm -f "$tmp"
        return 1
    fi

    mv -f "$tmp" "$dest"
    echo "✅ Download complete: $dest ($size bytes)"
}

mkdir -p "$MODEL_DIR"
if [ ! -f "$MODEL_DIR/$MODEL_FILE" ]; then
    echo "Downloading Flux.1 Dev model..."
    download_file "$MODEL_URL" "$MODEL_DIR/$MODEL_FILE" || exit 1
else
    echo "Flux.1 Dev model already exists. Skipping download."
fi

mkdir -p "$REPO_DIR/models/text_encoder"
if [ ! -f "$REPO_DIR/models/text_encoder/clip_l.safetensors" ]; then
    echo "Downloading clip_l.safetensors..."
    download_file "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors" "$REPO_DIR/models/text_encoder/clip_l.safetensors" || exit 1
fi
if [ ! -f "$REPO_DIR/models/text_encoder/t5xxl_fp8_e4m3fn.safetensors" ]; then
    echo "Downloading t5xxl_fp8_e4m3fn.safetensors..."
    download_file "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors" "$REPO_DIR/models/text_encoder/t5xxl_fp8_e4m3fn.safetensors" || exit 1
fi

mkdir -p "$REPO_DIR/models/VAE"
if [ ! -f "$REPO_DIR/models/VAE/ae.safetensors" ]; then
    echo "Downloading Flux VAE..."
    download_file "https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/ae.safetensors" "$REPO_DIR/models/VAE/ae.safetensors" || exit 1
fi

# 6. Check for existing WebUI instances and close them
echo "Checking for existing WebUI processes..."
pkill -f "launch.py" || true
sleep 2

# 7. Launch WebUI
echo "=========================================="
echo "Starting SD WebUI Forge Classic..."
echo "=========================================="

cd "$REPO_DIR"

# You can set environment variables for launch parameters here
# NOTE: Disable cuda-malloc and custom allocator flags to avoid CUDA init issues
# export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export COMMANDLINE_ARGS="--listen --port 7860 --xformers"
export PYTHON="$REPO_DIR/.venv/bin/python"

# Launch forge
$PYTHON launch.py
