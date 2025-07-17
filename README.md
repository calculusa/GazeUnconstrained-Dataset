# GazeUnconstrainedPipeline

An end-to-end AI engineering pipeline for unconstrained gaze estimation—from data collection to model training and evaluation. This repository showcases a complete system including multi-modal data collection, synchronized preprocessing, unified dataset loaders, multiple baseline models, and performance evaluation.

## 🔧 Project Structure

```
GazeUnconstrainedPipeline/
├── data/                # Sample data and download instructions
├── scripts/             # Data collection and preprocessing
├── datasets/            # Unified PyTorch Dataset class
├── models/              # Baseline model implementations
├── training/            # Training, evaluation, configuration
├── notebooks/           # Analysis and visualization
├── requirements.txt     # Python dependencies
├── .gitignore           # Git exclusions
├── LICENSE              # MIT License
└── README.md            # This documentation
```

## 📦 Features

- Multi-modal gaze dataset collection pipeline (Tobii + Webcam + Screen)
- Unified Dataset for iTracker, GazeTR, AFF-Net, Gaze360, and GazeNet
- Leave-one-subject-out training strategy
- Evaluation metrics: Angular Error and Euclidean Distance
- Visualization support (e.g., gaze heatmaps)

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/yourusername/GazeUnconstrainedPipeline.git
cd GazeUnconstrainedPipeline

# Install dependencies
pip install -r requirements.txt
```

## 🧪 Training

```bash
python training/train.py --model gazeTR --config training/config.yaml
```

## 📊 Sample Results

| Model    | Angular Error (°) | Euclidean Error (px) |
|----------|-------------------|----------------------|
| GazeTR   | 4.57              | 48.2                 |
| iTracker | 5.21              | 62.7                 |

## 🔒 Data

Only sample data is included. Full dataset available upon request:
> Contact: wen.zhou@xxx.ac.uk

## 📄 License

MIT License. See `LICENSE` for details.

## 🙏 Citation

If you use this project or dataset, please cite:

> Zhou et al., *GazeUnconstrained: A Dataset for Gaze Estimation in Unconstrained Environment*, 2025
