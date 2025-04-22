# Drawing with LLM 🎨

A Streamlit application that converts text descriptions into SVG graphics using multiple AI models.

## Overview

This project allows users to create vector graphics (SVG) from text descriptions using three different approaches:
1. **ML Model** - Uses Stable Diffusion to generate images and vtracer to convert them to SVG
2. **DL Model** - Uses Stable Diffusion for initial image creation and StarVector for direct image-to-SVG conversion
3. **Naive Model** - Uses Phi-4 LLM to directly generate SVG code from text descriptions

## Features

- Text-to-SVG generation with three different model approaches
- Adjustable parameters for each model type
- Real-time SVG preview and code display
- SVG download functionality
- GPU acceleration for faster generation

## Requirements

- Python 3.11+
- CUDA-compatible GPU (recommended)
- Dependencies listed in `requirements.txt`

## Installation

### Using Miniconda (Recommended)

```bash
# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Create and activate environment
conda create -n svg-app python=3.11 -y
conda activate svg-app

# Install star-vector
cd star-vector 
pip install -e .
cd ..

# Install other dependencies
pip install -r requirements.txt
```

### Using Docker

```bash
# Build and run with Docker Compose
docker-compose up -d
```

## Usage

Start the Streamlit application:

```bash
streamlit run app.py
```

Or with the yes flag to automatically accept:

```bash
yes | streamlit run app.py
```

The application will be available at http://localhost:8501

## Models

### ML Model (vtracer)
Uses Stable Diffusion to generate an image from the text prompt, then applies vtracer to convert the raster image to SVG.

Configurable parameters:
- Simplify SVG
- Color Precision
- Filter Speckle
- Path Precision

### DL Model (starvector)
Uses Stable Diffusion for initial image creation followed by StarVector, a specialized model designed to convert images directly to SVG.

### Naive Model (phi-4)
Directly generates SVG code using the Phi-4 language model with specialized prompting.

Configurable parameters:
- Max New Tokens

## Evaluation Data and Results

### Data
The `data` directory contains synthetic evaluation data created using custom scripts:
- The first 15 examples are from the Kaggle competition "Drawing with LLM"
- `descriptions.csv` - Text descriptions for generating SVGs
- `eval.csv` - Evaluation metrics
- `gen_descriptions.py` - Script for generating synthetic descriptions
- `gen_vqa.py` - Script for generating visual question answering data
- Sample images (`gray_coat.png`, `purple_forest.png`) for reference

### Results
The `results` directory contains evaluation results comparing different models:
- Evaluation results for both Naive (Phi-4) and ML (vtracer) models
- The DL model (StarVector) was not evaluated as it typically fails on transforming natural images, often returning blank SVGs
- Performance visualizations:
  - `category_radar.png` - Performance comparison across categories
  - `complexity_performance.png` - Performance relative to prompt complexity
  - `quality_vs_time.png` - Quality-time tradeoff analysis
  - `generation_time.png` - Comparison of generation times
  - `model_comparison.png` - Overall model performance comparison
- Generated SVGs and PNGs in respective subdirectories
- Detailed results in JSON and CSV formats

## Project Structure

```
drawing-with-llm/             # Root directory
│
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── Dockerfile                # Docker container definition
├── docker-compose.yml        # Docker Compose configuration
│
├── ml.py                     # ML model implementation (vtracer approach)
├── dl.py                     # DL model implementation (StarVector approach)
├── naive.py                  # Naive model implementation (Phi-4 approach)
├── gen_image.py              # Common image generation using Stable Diffusion
│
├── eval.py                   # Evaluation script for model comparison
├── eval_analysis.py          # Analysis script for evaluation results
├── metric.py                 # Metrics implementation for evaluation
│
├── data/                     # Evaluation data directory
│   ├── descriptions.csv      # Text descriptions for evaluation
│   ├── eval.csv              # Evaluation metrics
│   ├── gen_descriptions.py   # Script for generating synthetic descriptions
│   ├── gen_vqa.py            # Script for generating VQA data
│   ├── gray_coat.png         # Sample image by GPT-4o
│   └── purple_forest.png     # Sample image by GPT-4o
│
├── results/                  # Evaluation results directory
│   ├── category_radar.png    # Performance comparison across categories
│   ├── complexity_performance.png # Performance by prompt complexity
│   ├── quality_vs_time.png   # Quality-time tradeoff analysis
│   ├── generation_time.png   # Comparison of generation times
│   ├── model_comparison.png  # Overall model performance comparison
│   ├── summary_*.csv         # Summary metrics in CSV format
│   ├── results_*.json        # Detailed results in JSON format
│   ├── svg/                  # Generated SVG outputs
│   └── png/                  # Generated PNG outputs
│
├── star-vector/              # StarVector dependency (installed locally)
└── starvector/               # StarVector Python package
```

## Acknowledgments

This project utilizes several key technologies:
- [Stable Diffusion](https://github.com/CompVis/stable-diffusion) for image generation
- [StarVector](https://github.com/joanrod/star-vector) for image-to-SVG conversion
- [vtracer](https://github.com/visioncortex/vtracer) for raster-to-vector conversion
- [Phi-4](https://huggingface.co/microsoft/phi-4) for text-to-SVG generation
- [Streamlit](https://streamlit.io/) for the web interface
