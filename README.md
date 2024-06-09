# baseline_models_eeg

This repository contains baseline models for EEG (electroencephalography) data analysis. It provides implementations of various machine learning models and tools for EEG signal processing and classification.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Repository Structure](#repository-structure)
- [Contributing](#contributing)
- [License](#license)

## Overview
The `baseline_models_eeg` project includes baseline models to facilitate EEG data analysis, aiming to provide a solid foundation for further research and development in EEG-based applications.

## Features
- Preprocessing of EEG signals
- Implementation of baseline machine learning models
- Evaluation scripts for model performance

## Requirements
- Python 3.x
- Required packages listed in `requirements.txt`

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/MiladSoleymani/baseline_models_eeg.git
    cd baseline_models_eeg
    ```
2. Create a virtual environment and install dependencies:
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

## Usage
To run the preprocessing and model training scripts, use the following commands:
```bash
python scripts/preprocess.py --config configs/preprocess_config.yaml
python scripts/train.py --config configs/train_config.yaml
```

## Repository Structure
- `baseline`: Contains baseline models and training scripts.
- `notebooks`: Jupyter notebooks for data exploration and model development.
- `scripts`: Preprocessing and utility scripts.
- `configs`: Configuration files for preprocessing and training.
- `.gitignore`: Specifies files to ignore in the repository.
- `requirements.txt`: Lists the dependencies required for the project.
- `README.md`: Project documentation.

## Contributing
We welcome contributions to improve `baseline_models_eeg`. Please fork the repository, create a new branch, and submit a pull request with your changes.

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.
