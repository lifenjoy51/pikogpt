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

# Run tests
./gradlew test

# Run specific test classes
./gradlew test --tests "ValueTest"
./gradlew test --tests "train.TrainerTest"
./gradlew test --tests "sample.SamplerTest"

# Run main entry points
./gradlew run --args="train"        # Start training
./gradlew run --args="sample"       # Generate text
./gradlew run --args="prepare_data" # Prepare data for training
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
   - `SimpleBPE.kt` - Byte Pair Encoding tokenization implementation
   - `StoriesBpePrep.kt` - Data preprocessing pipeline (main entry point)
   - `StoryGenerator.kt` - External LLM integration for story generation
   - `MetaInfo.kt` - Vocabulary metadata structure

### Key Design Patterns

- **Pure Kotlin Implementation**: No external ML libraries (PyTorch, TensorFlow)
- **Automatic Differentiation**: Custom Value class tracks gradients through computation graph
- **Serialization**: Uses kotlinx.serialization for checkpoints and configuration
- **Modular Architecture**: Clear separation between model, training, and sampling components

## Training Workflow

1. **Data Preparation**: Run `StoriesBpePrep.kt` to tokenize text and create `data/[dataset]/train.bin` and `val.bin`
2. **Model Training**: `TrainerTest.kt` contains main entry points with predefined configurations (train_1k, etc.)
3. **Checkpointing**: Model state saved to `model/[steps]/[epoch]/checkpoint.json` and `model_weights.bin`
4. **Text Generation**: `Sampler.kt` loads checkpoints and generates text with configurable sampling strategies

## Main Entry Points

- `src/main/kotlin/data/StoriesBpePrep.kt` - Data preprocessing with BPE tokenization
- `src/main/kotlin/data/StoryGenerator.kt` - External LLM story generation
- `src/test/kotlin/train/TrainerTest.kt` - Training configurations and execution
- `src/test/kotlin/sample/SamplerTest.kt` - Text generation examples

## Configuration

- Training parameters in `TrainConfig.kt` (learning rate, batch size, model dimensions)
- Model architecture in `GPTConfig.kt` (layers, heads, embedding size)
- Sampling parameters in `SampleConfig.kt` (temperature, top-k, max tokens)

## Data Structure

```
data/
├── [dataset_name]/           # e.g., 1k, 3old, 6old
│   ├── stories.txt           # Original text input
│   ├── meta.json             # Vocabulary metadata (vocab size, mappings)
│   ├── train.bin             # Binary training data (tokenized)
│   ├── val.bin               # Binary validation data (tokenized)
│   └── unique_words.txt      # Unique vocabulary list

model/
├── [parameter_size]/                  # Model parameter size
│   └── [validation_loss]/              # validation loss * 10 at checkpoint
│       ├── checkpoint.json   # Model training state
│       ├── meta.json         # Model metadata
│       └── model_weights.bin # Serialized model parameters
```

## Key Files to Understand

- `Value.kt` - Automatic differentiation foundation (scalar autodiff engine)
- `gpt/PikoGPT.kt` - Main transformer model architecture
- `gpt/GPTConfig.kt` - Model configuration and hyperparameters
- `train/Trainer.kt` - Training loop with gradient accumulation and checkpointing
- `train/TrainConfig.kt` - Training hyperparameters and file paths
- `train/AdamW.kt` - AdamW optimizer implementation
- `sample/Sampler.kt` - Text generation with temperature and top-k sampling
- `data/SimpleBPE.kt` - Byte Pair Encoding tokenization
- `train/Checkpoint.kt` - Model state serialization for saving/loading

## Development Notes

- Tests are located in `src/test/kotlin/` and mirror the main source structure
- Main execution happens through test files rather than traditional main() methods
- The project uses kotlinx.serialization for JSON serialization of configs and checkpoints
- Model checkpoints include both model weights and optimizer state for resuming training
- The Value class implements a complete autodiff engine with gradient computation graph
- No external ML libraries are used - everything is implemented from scratch in Kotlin
