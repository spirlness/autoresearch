# Technology Stack: autoresearch

## Programming Language
- **Python (3.10+)**: Primary language for training, evaluation, and overall project logic.

## Training & Model Libraries
- **PyTorch**: Core deep learning framework for defining and training models.
- **Flash Attention**: Integrated for highly efficient memory usage and faster training.
- **torch.compile**: Utilized for optimizing PyTorch models for maximum training throughput.

## Testing & Quality Assurance
- **pytest**: Primary testing framework for unit and integration tests.
- **pytest-mock**: Plugin for mocking dependencies in tests.

## Package & Project Management
- **uv**: Modern, extremely fast Python package and project manager for dependency resolution and environment management.

## Infrastructure & Compute
- **Single NVIDIA GPU**: Optimized for high-performance training on a single NVIDIA GPU (e.g., H100, RTX 3060).

## Architecture & Data Management
- **Modular Monolith**: Organized into specialized modules (e.g., `autoresearch_trainer/`) for model, optimizer, and training logic.
- **mmap-based Token Cache**: Uses memory-mapped files for efficient, high-speed token loading during training.
