# 🛡️ ComfyUI NSFW Checker

> A powerful ComfyUI custom node for detecting NSFW (Not Safe For Work) content in both text and images using state-of-the-art machine learning models.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Compatible-green.svg)](https://github.com/comfyanonymous/ComfyUI)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## ✨ Features

- 🔤 **Text NSFW Detection** - Analyze text content for NSFW material using transformer-based models
- 🖼️ **Image NSFW Detection** - Detect NSFW content in images using vision transformer models
- 🎬 **Video Frame Analysis** - Support for analyzing video frames with configurable sampling intervals
- ⚡ **Batch Processing** - Efficient batch processing for multiple inputs
- 🎯 **Configurable Thresholds** - Adjustable confidence thresholds for NSFW detection
- 🚀 **GPU/CPU Support** - Choose between CUDA and CPU processing

## 🚀 Quick Start

### 📦 Installation

1. **Clone the repository** into your ComfyUI custom nodes directory:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/your-repo/ComfyUI-NSFW-Checker.git
```

2. **Install dependencies**:
```bash
cd ComfyUI-NSFW-Checker
pip install -r requirements.txt
```

3. **Restart ComfyUI** to load the new nodes.

## 🔧 Available Nodes

### 🔤 Load NSFW Text Model
> Loads a pre-trained text classification model for NSFW detection.

| Input | Type | Description | Options |
|-------|------|-------------|---------|
| `model` | String | Text classification model | `michellejieli/NSFW_text_classifier` |
| `device` | String | Processing device | `cuda`, `cpu` |

| Output | Type | Description |
|--------|------|-------------|
| `model` | Pipeline | Loaded text classification pipeline |

### 🖼️ Load NSFW Vision Model
> Loads a pre-trained image classification model for NSFW detection.

| Input | Type | Description | Options |
|-------|------|-------------|---------|
| `model` | String | Vision classification model | `AdamCodd/vit-base-nsfw-detector`, `Falconsai/nsfw_image_detection` |
| `device` | String | Processing device | `cuda`, `cpu` |

| Output | Type | Description |
|--------|------|-------------|
| `model` | Pipeline | Loaded image classification pipeline |

### 📝 Check NSFW Text
> Analyzes text content for NSFW material.

| Input | Type | Description | Default |
|-------|------|-------------|---------|
| `model` | Pipeline | Text classification pipeline | Required |
| `text` | String | Text content to analyze | Required |
| `threshold` | Float | Confidence threshold (0.0-1.0) | `0.5` |

| Output | Type | Description |
|--------|------|-------------|
| `nsfw` | Boolean | Whether content is NSFW |
| `nsfw_score` | Float | Confidence score (0.0-1.0) |

### 🎬 Check NSFW Vision
> Analyzes images for NSFW content.

| Input | Type | Description | Default |
|-------|------|-------------|---------|
| `model` | Pipeline | Image classification pipeline | Required |
| `image` | Tensor | Image tensor to analyze | Required |
| `interval` | Integer | Frame sampling interval (0 = all frames) | `0` |
| `threshold` | Float | Confidence threshold (0.0-1.0) | `0.5` |

| Output | Type | Description |
|--------|------|-------------|
| `nsfw` | Boolean | Whether content is NSFW |
| `nsfw_score` | Float | Average confidence score across frames |

## 🤖 Model Information

### 📝 Text Models
| Model | Description | Performance |
|-------|-------------|-------------|
| `michellejieli/NSFW_text_classifier` | Transformer-based model for NSFW text classification | High accuracy |

### 🖼️ Vision Models
| Model | Description | Performance |
|-------|-------------|-------------|
| `AdamCodd/vit-base-nsfw-detector` | Vision Transformer for NSFW image detection | High accuracy |
| `Falconsai/nsfw_image_detection` | Alternative NSFW image detection model | Good accuracy |

## 🤝 Contributing

We welcome contributions! Here's how you can help:

- 🐛 **Report bugs** and suggest features
- 🔧 **Submit pull requests** for improvements
- 📖 **Improve documentation** and examples
- 🧪 **Test** on different systems and configurations

### Made with ❤️ for the ComfyUI community