#!/bin/bash
# Pod Startup Script for SD WebUI Forge Classic (Neo) + Flux.1 Dev

set -e

WORKSPACE="/workspace"
REPO_DIR="$WORKSPACE/sd-webui-forge-classic"
REPO_URL="https://github.com/nolanlane/sd-webui-forge-classic.git"
BRANCH="neo"
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
    export PATH="$HOME/.cargo/bin:$PATH"
else
    echo "uv package manager is already installed."
fi

# 3. Python 3.13 via uv
if ! uv python list | grep -q '3.13'; then
    echo "Installing Python 3.13 via uv..."
    uv python install 3.13
else
    echo "Python 3.13 is already available via uv."
fi

# 4. Clone Repository
if [ ! -d "$REPO_DIR" ]; then
    echo "Cloning Forge Classic (Neo branch) into $WORKSPACE..."
    git clone --branch $BRANCH $REPO_URL $REPO_DIR
else
    echo "Repository already exists at $REPO_DIR. Checking for updates..."
    # Optionally update the repo: cd $REPO_DIR && git pull origin $BRANCH
fi

# 5. Download Flux.1 Dev Model
mkdir -p "$MODEL_DIR"
if [ ! -f "$MODEL_DIR/$MODEL_FILE" ]; then
    echo "Downloading Flux.1 Dev model..."
    curl -L "$MODEL_URL" -o "$MODEL_DIR/$MODEL_FILE"
else
    echo "Flux.1 Dev model already exists. Skipping download."
fi

# 6. Launch WebUI
echo "=========================================="
echo "Starting SD WebUI Forge Classic..."
echo "=========================================="

cd "$REPO_DIR"

# Ensure uv is used by webui.sh
# You can set environment variables for launch parameters here
export COMMANDLINE_ARGS="--listen --port 7860 --xformers --uv-symlink"
export PYTHON="uv run --python 3.13 python3"

# Launch forge
bash webui.sh
