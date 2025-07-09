# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PikoGPT is a Kotlin port of nanoGPT and micrograd by Andrej Karpathy. This is an educational project that implements a GPT model with automatic differentiation from scratch in Kotlin, without external ML libraries.

## Build and Development Commands

```bash
# Build the project
./gradlew build

# Clean build artifacts
./gradlew clean
```

## Architecture

### Core Components

1. **Value Class (`Value.kt`)** - Automatic differentiation engine
   - Implements scalar autodiff with gradient tracking
   - Supports basic operations (+, -, *, /, pow) and activations (ReLU, GELU, sigmoid)
   - Core backward() method for gradient computation

2. **GPT Model (`src/main/kotlin/gpt/`)**
   - `PikoGPT.kt` - Main transformer model
   - `GPTConfig.kt` - Model configuration (layers, heads, embedding dimensions)
   - `TransformerBlock.kt` - Individual transformer blocks
   - `SimpleSelfAttention.kt` - Self-attention mechanism
   - `FeedForward.kt` - MLP component
   - `LayerNorm.kt` - Layer normalization
   - `Linear.kt` - Linear transformation layer

3. **Training System (`src/main/kotlin/train/`)**
   - `Trainer.kt` - Main training loop with checkpointing
   - `TrainConfig.kt` - Training hyperparameters
   - `AdamW.kt` - AdamW optimizer implementation
   - `DataLoader.kt` - Batch loading for training
   - `Checkpoint.kt` - Model state serialization

4. **Sampling (`src/main/kotlin/sample/`)**
   - `Sampler.kt` - Text generation with temperature and top-k sampling
   - `SampleConfig.kt` - Sampling parameters

5. **Data Processing (`src/main/kotlin/data/`)**
   - `DataLoader.kt` - Training data loading
   - `MetaInfo.kt` - Vocabulary metadata
   - Character and BPE tokenization implementations

### Key Design Patterns

- **Pure Kotlin Implementation**: No external ML libraries (PyTorch, TensorFlow)
- **Automatic Differentiation**: Custom Value class tracks gradients through computation graph
- **Serialization**: Uses kotlinx.serialization for checkpoints and configuration
- **Modular Architecture**: Clear separation between model, training, and sampling components

## Training Workflow

1. **Data Preparation**: Text is tokenized and stored in `data/[dataset]/train.bin` and `val.bin`
2. **Model Training**: `Trainer` class handles training loop with gradient accumulation
3. **Checkpointing**: Model state saved to `out/[dataset]/checkpoint.json` and `model_weights.bin`
4. **Text Generation**: `Sampler` loads checkpoints and generates text

## Configuration

- Training parameters in `TrainConfig.kt` (learning rate, batch size, model dimensions)
- Model architecture in `GPTConfig.kt` (layers, heads, embedding size)
- Sampling parameters in `SampleConfig.kt` (temperature, top-k, max tokens)

## Data Structure

```
data/
├── shakespeare_char/
│   ├── input.txt      # Original text
│   ├── meta.json      # Vocabulary info
│   ├── train.bin      # Training data
│   └── val.bin        # Validation data
```

## Key Files to Understand

- `Value.kt` - Automatic differentiation foundation
- `gpt/PikoGPT.kt` - Main model architecture
- `train/Trainer.kt` - Training loop and checkpointing
- `sample/Sampler.kt` - Text generation
- `train/TrainConfig.kt` - Default hyperparameters
