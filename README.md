# Vision Transformer (ViT) - CIFAR-100 Classification

A comprehensive deep learning project implementing Vision Transformer architectures for image classification on CIFAR-100. This project compares ViT trained from scratch, ViT with transfer learning using PyTorch, and MobileNetV3 transfer learning with TensorFlow. Includes an interactive Streamlit GUI for inference.

## Project Overview

This project explores Vision Transformers (ViT), a groundbreaking architecture that applies transformer models from NLP to computer vision. Rather than using convolutional operations, ViT treats images as sequences of patches, enabling pure transformer-based image classification.

### Key Features
- **Multiple Implementations**: ViT from scratch, ViT transfer learning (PyTorch), and MobileNetV3 transfer learning (TensorFlow)
- **Transformer Architecture**: Standard multi-head self-attention without image-specific inductive biases
- **Patch-Based Processing**: Images converted to patches and treated as tokens
- **Transfer Learning**: Leverages pretrained models for better accuracy
- **Framework Comparison**: PyTorch and TensorFlow implementations
- **Streamlit GUI**: User-friendly interface for real-time image classification
- **Performance Analysis**: Comparison of training strategies

## Dataset

Uses CIFAR-100 dataset containing 100 object categories with 60,000 images (32×32 pixels). Due to hardware limitations, models trained for only 1 epoch, making transfer learning significantly more effective than training from scratch.

## Architecture: Vision Transformer (ViT)

Vision Transformers revolutionize computer vision by replacing convolutional operations with pure transformer architecture, eliminating image-specific inductive biases except for initial patch extraction.

**Key Innovation**: Converting images into patches and treating them as tokens, similar to words in NLP.

**Architecture Components**:

**Patch Embedding**: Images are divided into small patches (e.g., 16×16 pixels). These patches are flattened and linearly projected to create patch embeddings, treating them as tokens in NLP.

**Positional Embeddings**: Learnable position embeddings are added to patch embeddings to preserve spatial information about patch locations in the image.

**Classification Token (CLS)**: A learnable classification token is prepended to the patch embeddings. This token aggregates information from all patches and is used for final classification.

**Multi-Head Self-Attention**: Projects patch embeddings as Query, Key, and Value vectors to calculate dynamic relevance between all image parts. Enables the model to focus on different features across the entire image simultaneously.

**Multi-Layer Perceptron (MLP) Block**: Two dense layers with GELU activation. First layer expands dimensions by a factor of 4 (e.g., 768 → 3072) to project features into richer space for learning complex non-linear interactions. Second layer contracts back to original dimensions.

**Transformer Encoder**: Stack of multi-head attention and MLP blocks. After passing through all encoder blocks, the CLS token is extracted as the feature representation of the entire image and passed to classification head for final prediction.

**No Convolutional Bias**: Unlike prior vision works, ViT does not introduce image-specific inductive biases into the architecture apart from patch extraction, allowing transformers to learn vision tasks from scratch.

## Project Approach

### Development Strategy

1. **Foundation Building**: Implemented ViT from scratch in PyTorch, building multi-head attention, MLP layers, and transformer blocks from ground up
2. **Transfer Learning (PyTorch)**: Used pretrained 'vit_tiny_patch16_224' model with custom classification head
3. **Transfer Learning (TensorFlow)**: Attempted ViT but faced compatibility issues, switched to MobileNetV3 which proved most effective
4. **GUI Development**: Built Streamlit interface using best-performing PyTorch transfer learning model

### Training Constraints and Results

Due to hardware limitations, all models trained for only 1 epoch:

**From Scratch (PyTorch ViT)**: Very low accuracy (~20%) - Training deep transformer architectures requires many epochs and large datasets to learn meaningful patterns.

**Transfer Learning PyTorch (vit_tiny)**: Better accuracy but still limited - Pretrained ViT weights helped, but 1 epoch insufficient for fine-tuning.

**Transfer Learning TensorFlow (MobileNetV3)**: Best accuracy - CNN-based architecture with pretrained ImageNet weights adapted faster to CIFAR-100 with limited epochs.

### Why Transfer Learning Dominated

Transfer learning proved dramatically more effective because:
- Pretrained models already learned general visual features from large datasets
- Only classification head needed adaptation for CIFAR-100
- Requires fewer training iterations and data to achieve good accuracy
- Overcomes single-epoch limitation through initialization from learned representations

## File Structure

- `ViT_CIFAR_100.ipynb` - ViT trained from scratch (PyTorch)
- `ViT_TL_CIFAR_100.ipynb` - ViT transfer learning (PyTorch with vit_tiny_patch16_224)
- `ViT_TL_TF_CIFAR_100.ipynb` - Transfer learning (TensorFlow with MobileNetV3)
- `gui/` - Streamlit GUI application
  - `app.py` - Interactive inference interface
- `requirements.txt` - Project dependencies
- `.gitignore` - Git ignore file
- `README.md` - This file

## Installation

### Requirements
- Python 3.8+
- PyTorch 2.5+
- TensorFlow 2.15+
- Streamlit
- timm (PyTorch Image Models for pretrained ViT)
- NumPy, Pandas, Matplotlib, Pillow

### Setup
```bash
# Clone repository
git clone https://github.com/mdarshanrshaikh/vision-transformer-cifar100.git
cd vision-transformer-cifar100

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Jupyter Notebooks

Run individual model implementations:

**ViT From Scratch:**
```bash
jupyter notebook ViT_CIFAR_100.ipynb
```

**ViT Transfer Learning (PyTorch):**
```bash
jupyter notebook ViT_TL_CIFAR_100.ipynb
```

**Transfer Learning (TensorFlow):**
```bash
jupyter notebook ViT_TL_TF_CIFAR_100.ipynb
```

### GUI Inference

Run the Streamlit application for interactive predictions:
```bash
streamlit run gui/app.py
```

The GUI allows you to:
- Upload images for classification
- Get model predictions with confidence scores
- Visualize top predictions
- Compare model performances

## Key Learnings

1. **Transformers Don't Need Convolutional Bias**: ViT proves that eliminating image-specific inductive biases and using pure transformers is viable for vision tasks when properly trained.

2. **Patch Embeddings Are Effective**: Converting images to patch sequences and treating them as tokens enables transformers to learn visual patterns without convolutions.

3. **Transfer Learning Necessity**: Training transformers from scratch requires massive datasets and computational resources. With limited epochs and data, pretrained models are essential.

4. **CNN Still Competitive**: Despite transformers being cutting-edge, well-designed CNNs (MobileNetV3) with transfer learning remain highly competitive and sometimes more efficient.

5. **Framework Trade-offs**: Pretrained model availability differs between frameworks. PyTorch had timm library for ViT; TensorFlow compatibility issues required alternative architecture.

6. **Patch Size Matters**: Smaller patches capture more detail but increase sequence length and computational cost. Trade-offs exist between granularity and efficiency.

7. **Single Epoch Limitation**: Transformers particularly suffer from limited training epochs compared to CNNs. Transfer learning becomes critical constraint mitigation strategy.

## Model Performance

**From Scratch (1 epoch)**:
- ViT PyTorch: ~20% accuracy - Demonstrates difficulty of training transformers without pretrained weights

**Transfer Learning (1 epoch)**:
- ViT PyTorch (vit_tiny): Better but still limited - Pretrained weights helped but insufficient epochs
- MobileNetV3 TensorFlow: Best accuracy - CNN adaptation faster than transformer fine-tuning

**Key Insight**: Transfer learning provided 3-5x better accuracy than from-scratch training, highlighting importance of pretrained representations for vision transformers.

## Architecture Comparison

**ViT Advantages**:
- No inductive biases except patch extraction
- Global receptive field from start (through self-attention)
- Scalable to larger models and datasets
- Potential for multimodal learning

**CNN Advantages** (MobileNetV3):
- Leverages local spatial structure through convolution
- More efficient with limited data and epochs
- Faster training and inference
- Better suited for resource-constrained scenarios

## References

**Paper**: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (Dosovitskiy et al., 2020)

**Key Concepts**:
- First successful application of pure transformer to vision
- Patch-based image tokenization
- Multi-head self-attention for global context
- Eliminating convolutional inductive biases
- Transfer learning importance for vision transformers

**Resources**:
- ViT Implementation Details: https://github.com/google-research/vision_transformer
- Timm Library: https://github.com/rwightman/pytorch-image-models
- CIFAR-100 Dataset: https://www.cs.toronto.edu/~kriz/cifar.html

## Contact

Questions or feedback? Feel free to open an issue or reach out.
