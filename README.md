# College Predictor

A simple CLI tool to predict college admissions based on historical data and machine learning.

## Motive
Help students estimate their chances of admission to various colleges using previous years' cut-off data and predictive modeling.

## Tech Stack
- Python 3
- Pandas
- Scikit-learn
- NumPy

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/thesaiprasadrao/jee-main-college-predictor.git
   cd jee-main-college-predictor
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the predictor:
   ```bash
   python predictor_interface.py
   ```
4. Model File
   https://drive.google.com/file/d/1JAXw64b99sqY9NuhfKT48JGN-5k241Ut/view?usp=sharing

## Project Structure
```
├── main.py                    # Main entry point for training and demo
├── predictor_interface.py     # Interactive CLI interface for predictions
├── ml_model.py               # Machine learning models and prediction logic
├── data_loader.py            # Data loading and preprocessing utilities
├── requirements.txt          # Python dependencies
├── trained_college_predictor.pkl  # Pre-trained ML model (generated)
└── data/                     # Historical admission data (2020-2024)
    ├── 2020/                 # Round-wise admission data for 2020
    ├── 2021/                 # Round-wise admission data for 2021
    ├── 2022/                 # Round-wise admission data for 2022
    ├── 2023/                 # Round-wise admission data for 2023
    └── 2024/                 # Round-wise admission data for 2024
```

## Data
- Historical cut-off data is stored in the `data/` folder.

## Usage
- Follow the prompts in the CLI to input your details and get predictions.
