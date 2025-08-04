# Wildfire-DNN: A Spatio-Temporal Deep Learning Framework for Wildfire Ignition Forecasting

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

This repository contains the official source code and manuscript for the research paper: *"A Unified Spatio-Temporal Deep Learning Framework for Probabilistic Wildfire Ignition Forecasting Using Satellite Data Fusion and 3D Canopy Metrics"*.

This project introduces **Wildfire-DNN**, a novel deep learning framework designed to provide a probabilistic forecast of wildfire ignition. It fuses heterogeneous data sources, including satellite imagery, meteorological reanalysis, and 3D vegetation metrics from NASA's GEDI mission.

![Model Architecture](https://storage.googleapis.com/generative-ai-downloads/images/fig_architecture.png)

## Repository Structure

```
Wildfire-DNN/
├── .gitignore          # Specifies files to be ignored by Git
├── LICENSE             # Project license (MIT)
├── README.md           # This guide
├── requirements.txt    # Python package dependencies
├── data/               # Placeholder for project data (not tracked by Git)
│   └── README.md       # Instructions for acquiring data
├── notebooks/          # Jupyter notebooks for exploration and prototyping
│   ├── 01_Data_Exploration.ipynb
│   └── 02_Model_Prototyping.ipynb
├── paper/              # LaTeX source for the research paper
│   ├── main.tex
│   ├── references.bib
│   └── figures/
├── src/                # Python source code
│   ├── data_preprocessing.py  # Scripts for data loading and cleaning
│   ├── model.py               # Wildfire-DNN model definition
│   ├── train.py               # Script to train the model
│   └── evaluate.py            # Script to evaluate the trained model
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/Wildfire-DNN.git
    cd Wildfire-DNN
    ```

2.  **Create a Python virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

## Data Acquisition

The raw data required for this project is not included in the repository due to its large size. You must download it manually from the official sources and place it in the `data/raw/` directory. See the guide in `data/README.md` for detailed instructions and links.

## Workflow

1.  **Preprocess Data:**
    Run the preprocessing script to convert the raw data into a format suitable for the model.
    ```bash
    python data/data_preprocessing.py --raw_dir data/raw --output_dir data/processed
    ```

2.  **Train the Model:**
    Train the Wildfire-DNN model using the processed data.
    ```bash
    python data/train.py --data_path data/processed/training_data.h5 --epochs 50 --batch_size 32
    ```

3.  **Evaluate the Model:**
    Evaluate the performance of the trained model on the test set. This will generate the metrics table and ROC curve plot.
    ```bash
    python data/evaluate.py --model_path models/wildfire_dnn.h5 --test_data data/processed/test_data.h5
    ```

## Building the Paper

To compile the LaTeX paper into a PDF, navigate to the `paper/` directory and use a LaTeX distribution (like TeX Live, MiKTeX).

```bash
cd paper/
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Citation

If you use this work, please cite the paper:

```bibtex
@article{Mamun2025WildfireDNN,
  title={{A Unified Spatio-Temporal Deep Learning Framework for Probabilistic Wildfire Ignition Forecasting Using Satellite Data Fusion and 3D Canopy Metrics}},
  author={Mamun, Mostafijur Rahman},
  journal={Journal of Geo-Information and AI},
  year={2025},
  publisher={Imaginary Publishing}
}
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
