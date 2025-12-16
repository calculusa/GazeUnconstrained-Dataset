# GazeUnconstrainedPipeline

An end-to-end AI engineering pipeline for unconstrained gaze estimation—from data collection to model training and evaluation. This repository showcases a complete system including multi-modal data collection, synchronized preprocessing, unified dataset loaders, multiple baseline models, and performance evaluation.

## 🔧 Project Structure

```
GazeUnconstrainedPipeline/
├── data/                # Sample data and download instructions
├── scripts/             # Data collection and preprocessing
├── trainTestonGazeUniconstrained/  # Unified PyTorch Dataset class
├                                   # Baseline model implementations
├                                   # Training, evaluation, configuration
├── dataVisualization/   # Analysis and visualization
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

| Model    | Angular Error (°) | Euclidean Error (cm) |
|----------|-------------------|----------------------|
| AFF-Net  | --                | 2.26                 |
| iTracker | --                | 2.43                 |
| GazeTR-Hybrid   | 6.58       | --                   |
| Full-Face| 4.57              | --                   |
| Gaze360  | 6.01              | --                   | 


## 🔒 Data

Only sample data is included. Full dataset available upon request:
> Contact: This information is hidden for the anonymous review process and will be disclosed once the review is complete.
## 📄 License

MIT License. See `LICENSE` for details.

## 🙏 Citation

If you use this project or dataset, please cite:


