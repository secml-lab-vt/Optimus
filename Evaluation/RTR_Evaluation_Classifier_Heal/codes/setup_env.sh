#!/bin/bash

echo "Creating conda environment for LlamaGuard evaluation..."

# Remove existing environment if it exists
conda env remove -n llama_guard_eval --all --yes 2>/dev/null || true

# Create the conda environment
echo "Installing conda environment..."
conda env create -f environment.yml

if [ $? -eq 0 ]; then
    echo "Environment created successfully!"
    
    # Activate the environment
    echo "Activating environment..."
    source activate llama_guard_eval
    
    # Verify installation
    echo "Verifying installation..."
    python -c "import torch; print(f'PyTorch version: {torch.__version__}')" 2>/dev/null || echo "PyTorch not found"
    python -c "import transformers; print(f'Transformers version: {transformers.__version__}')" 2>/dev/null || echo "Transformers not found"
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" 2>/dev/null || echo "CUDA check failed"
    
    echo "Environment setup complete!"
    echo "To activate the environment in the future, run:"
    echo "conda activate llama_guard_eval"
else
    echo "Failed to create environment. Trying alternative approach..."
    echo "Creating environment manually..."
    
    conda create -n llama_guard_eval python=3.10 -y
    conda activate llama_guard_eval
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
    pip install transformers==4.51.0 datasets scikit-learn pandas numpy tqdm wandb matplotlib seaborn accelerate bitsandbytes sentencepiece protobuf huggingface_hub safetensors
    
    echo "Manual installation complete!"
fi 