# VC-SD Reproduction Repository

This repository contains the most important reproduction scripts for the **VC-SD** paper.

## Setup Instructions

### 1. Create a Conda Environment

It is recommended to use **Python 3.9**:

```bash
conda create -n vcsd python=3.9
conda activate vcsd
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Demonstration

If you just want to play with the model itself, this can be done in demo.ipynb. 
There is no need to download the data for this stage as demonstration-data is provided.

### 4. Download Required Datasets

```bash
bash ./download_data.sh
```

### 5. Create Validation Dataset

```bash
python create_data.py
```

### 6. Run Reproduction Scripts Audio

All paths should be reconfigured, but can be set in each individual file.

```bash
python scripts/plot_content_comparison.py
python scripts/plot_prosody_output.py
python scripts/convert.py
python scripts/get_metrics.py
```

### 7. Run Reproduction Scripts Evaluation

All .csv file are included. Paths should be reconfigured.

```bash
python evaluation/pitch_ab.py
python evaluation/nMOS_seen.py
python evaluation/nMOS_unseen.py 
```
