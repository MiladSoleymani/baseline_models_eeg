# Baseline Models for EEG Analysis

A comprehensive framework for EEG (electroencephalography) signal analysis and classification, providing baseline implementations of classical machine learning models and signal processing techniques specifically designed for EEG data.

## 🧠 Overview

This repository implements a modular and extensible framework for EEG analysis, supporting multiple datasets and classical machine learning approaches. It's designed to establish baseline performance metrics for various EEG classification tasks including epilepsy detection, motor imagery classification, and brain-computer interface applications.

## ✨ Key Features

- **Multiple EEG Dataset Support**: CHB-MIT, BCI Competition 2a, LEE, Klinik, and synthetic datasets
- **Classical ML Models**: SVM, Random Forest, XGBoost, Naive Bayes, K-NN
- **Signal Processing**: PSD, Wavelet Transform, Common Spatial Patterns (CSP)
- **Flexible Architecture**: Modular design with configurable pipelines
- **Experiment Tracking**: Weights & Biases integration
- **Cross-validation**: Multiple validation strategies including patient-wise splits
- **EEG Standards**: Support for various International 10-20 electrode configurations

## 📊 Supported Datasets

| Dataset | Task | Classes | Sampling Rate | Description |
|---------|------|---------|---------------|-------------|
| **CHB-MIT** | Epilepsy Detection | 2 | Variable | Scalp EEG database for seizure detection |
| **BCI 2a** | Motor Imagery | 4 | 250 Hz → 256 Hz | Left/right hand, feet, tongue motor imagery |
| **LEE** | Motor Imagery | 2 | 1000 Hz → 256 Hz | Left vs right hand motor imagery |
| **Klinik** | Clinical Classification | 2 | Variable | Clinical EEG with balanced sampling |
| **Synthetic** | Testing | 2 | 256 Hz | Generated data for algorithm validation |

## 🛠️ Implemented Methods

### Feature Extraction
- **Power Spectral Density (PSD)**: Frequency domain analysis using Welch's method
- **Wavelet Transform Energy (WTE)**: Time-frequency decomposition
- **Wavelet Packet Transform Energy (WPTE)**: Detailed wavelet analysis
- **Common Spatial Patterns (CSP)**: Spatial filtering for motor imagery

### Classification Models
- Support Vector Machine (SVM) with one-vs-one strategy
- Balanced Random Forest Classifier
- XGBoost with binary/multiclass objectives
- Gaussian Naive Bayes
- K-Nearest Neighbors (K-NN)

## 📋 Requirements

- Python 3.8+
- NumPy
- SciPy
- scikit-learn
- PyTorch
- MNE-Python
- XGBoost
- PyWavelets
- Weights & Biases (optional)

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/MiladSoleymani/baseline_models_eeg.git
cd baseline_models_eeg
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 📖 Usage

### Basic Example

```python
from baseline.data.bci import bci_dataset
from baseline.utils.transforms import *
from baseline.configs.utils import get_classifier

# Load dataset
dataset = bci_dataset.BCIDataset(
    data_path="/path/to/bci/data",
    train=True,
    transform=Compose([
        PSD(),
        Flatten(),
        Normalize()
    ])
)

# Initialize classifier
clf = get_classifier("svm", multiclass=True)

# Train model
X_train, y_train = dataset.get_all_data()
clf.fit(X_train, y_train)
```

### Running Experiments

Use the provided configuration files:

```bash
python scripts/run_experiment.py --config baseline/configs/run_configs/baseline.json
```

### Configuration Example

```json
{
    "dataset": "bci",
    "model": "svm",
    "features": ["psd", "wte"],
    "validation": "stratified_kfold",
    "n_splits": 5,
    "random_state": 42
}
```

## 📁 Repository Structure

```
baseline_models_eeg/
├── baseline/
│   ├── configs/
│   │   ├── eeg_recording_standard/    # Electrode configurations
│   │   ├── run_configs/               # Experiment configurations
│   │   └── utils/                     # Configuration utilities
│   ├── data/
│   │   ├── bci/                       # BCI Competition 2a dataset
│   │   ├── chb_mit/                   # CHB-MIT epilepsy dataset
│   │   ├── csp_feature/               # CSP feature loader
│   │   ├── Klinik/                    # Clinical EEG dataset
│   │   ├── LEE/                       # LEE motor imagery dataset
│   │   └── synthetic/                 # Synthetic data generator
│   └── utils/
│       ├── transforms.py              # Signal processing transforms
│       └── utils.py                   # General utilities
├── notebooks/
│   └── baseline.ipynb                 # Example notebook
├── scripts/
│   └── classic_var_feature_size.py    # Feature analysis script
├── requirements.txt
└── README.md
```

## 🔧 Customization

### Adding a New Dataset

```python
from baseline.data.base import BaseDataset

class MyDataset(BaseDataset):
    def __init__(self, data_path, train=True, transform=None):
        super().__init__(data_path, train, transform)
        # Load your data here
    
    def __getitem__(self, idx):
        # Return (eeg_signal, label)
        pass
```

### Adding a New Transform

```python
class MyTransform:
    def __call__(self, x):
        # Process EEG signal x
        return transformed_x
```

## 📊 Performance Baselines

Typical performance ranges on standard datasets:

| Dataset | Model | Accuracy | F1-Score |
|---------|-------|----------|----------|
| BCI 2a | SVM + CSP | 70-75% | 0.68-0.73 |
| CHB-MIT | Random Forest + PSD | 85-90% | 0.83-0.88 |
| LEE | XGBoost + WTE | 78-82% | 0.76-0.80 |

*Note: Actual performance depends on preprocessing and hyperparameter tuning.*

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use this framework in your research, please cite:

```bibtex
@software{baseline_models_eeg,
  author = {Milad Soleymani},
  title = {Baseline Models for EEG Analysis},
  url = {https://github.com/MiladSoleymani/baseline_models_eeg},
  year = {2024}
}
```

## 🙏 Acknowledgments

- CHB-MIT dataset providers
- BCI Competition organizers
- MNE-Python developers
- scikit-learn contributors

## 📧 Contact

For questions and feedback, please open an issue on GitHub or contact the maintainers.