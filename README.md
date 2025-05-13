# STAT-4830-Group-Project
## Group Members: Shuhan Zhang, Xinyu Zhang, Lexuan Chen

## Summary

**Problem Statement**  
This project tackles the problem of predicting consumer hotel preferences from personalized offer sets, using real-world data from Expedia. The goal is to model and anticipate user choices based on both hotel attributes and consumer features.

**Modeling Approaches**  
Two complementary frameworks were developed:
- A **mixture preference learning model**, optimized via the **Frank-Wolfe algorithm**, with extensions to neural networks for capturing nonlinear utility signals.
- A **matrix completion approach** based on **low-rank factorization**, designed to recover missing preference data and improve predictive accuracy.

**Key Findings & Contributions**
- **Linear models** consistently outperformed neural networks, likely due to the limited dataset size and the relatively linear nature of consumer decision processes.
- The **matrix completion framework** showed strong robustness, particularly when enhanced with **bias-aware initialization** and **Huber loss**, which together helped mitigate data sparsity and noise.
- The study offers practical modeling strategies for preference prediction, and demonstrates how classical optimization and matrix recovery methods can be effectively applied in **large-scale recommendation systems**.



## Repository Structure Overview

```
.
├── src/
│   └── Core implementation files, including:
│       - frank_wolfe.py
│       - adam_expedia_completion.py
│       These define key classes (e.g., Problem_FrankWolfe) and functions for model training and evaluation.
│
├── notebooks/
│   └── Cleaned Jupyter notebooks demonstrating:
│       - Frank-Wolfe Algorithm
│       - Matrix Completion pipelines
│       Each notebook imports from `src/` and walks through preprocessing, training, and visualization.
│       A readme file summarizes the purpose and content of each demo.
│
├── tests/
│   └── Test scripts and datasets, including:
│       - train_28-32_10000.csv
│       - test_28-32_1000.csv
│       And other necessary datasets. These are used to verify correctness and evaluate performance.
│
├── docs/
│   └── Final presentation slides.
│
├── report.md
│   └── Final project report covering:
│       - Background
│       - Modeling methodology
│       - Experimental results
│       - Key takeaways
│
├── requirements.txt
│   └── Python package dependencies (e.g., torch, numpy, pandas) for consistent reproducibility.
│
├── development_history/
│   └── Archived files from earlier stages, including:
│       - Draft code and notebooks
│       - Interim reports and slides (with self-critiques)
│       - Figures and exploratory LLM-related work
```


## Setup Instructions

We **highly recommend using [Google Colab](https://colab.research.google.com/)** to view and run our final demo. It automatically handles all dependencies and environment setup.  
Simply open the notebook and click:

```
Runtime → Run all
```

The full pipeline will execute, including downloading data, model training, and result visualization.

### Running Locally

If you'd like to run the code on your local machine, follow these steps:

1. **Prepare a Jupyter Environment**  
   Make sure [Jupyter Notebook](https://jupyter.org/) is installed and configured with a Python **3.9+** kernel.
2. **Install Dependencies**  
   Use the provided `requirements.txt` to install all necessary packages:
   ```bash
   pip install -r requirements.txt
   ```
3. **Download Source Files**  
   Make sure you have the following files in your local directory:
   - The notebook: `notebooks/Demo.ipynb`
   - Source code in `src/`
   - Any additional scripts or test files
4. **Set the Python Path (if needed)**  
   Ensure your environment can locate the `src/` module:
   ```python
   import sys
   sys.path.append('../src')
   ```If needed, add the project root to your PYTHONPATH or append manually in the notebook.
5. **Run the Notebook**  
   Open and execute Demos in `notebooks/`:
   - It will import functions from `src/`
   - Train the model and display results
   - Include convergence plots, evaluation metrics, and ranking tables

### Dataset Access

All datasets required by the demo are **hosted in the `test/` folder of our GitHub repository**, and are **automatically loaded via URL** in the notebook.  
There is no need for manual download or local storage.

### GPU Support (Colab)

Our entire pipeline is compatible with **Google Colab GPU runtime**, and runs on **T4 GPUs** by default.

Training and evaluation steps have been optimized for T4 hardware.  
You’ll benefit from accelerated matrix operations, especially during SVD and model fitting phases.


## Running the Code
The entire final results can be executed seamlessly in Google Colab with one click.
Our final notebook automatically imports all core functions from the src/ directory and performs end-to-end training, evaluation, and visualization. Unit tests from the tests/ folder can also be executed within the notebook. 
To run the project: 👉 Click “Runtime → Run all” in Colab — everything will install and execute automatically, no manual setup required.
For more details and a breakdown of each step, see the Colab demo section.

## Executable Demo Link

## Frank-Wolfe Algorithm (MNL and NN-based)
[https://colab.research.google.com/drive/1u8P8WUMnW8a9PnYXDeWyapwQbmQUbVnf?usp=sharing](https://colab.research.google.com/drive/14N6OsXc01ejDGrXR_4BklPs-PuP9zc4L?usp=sharing)

## Matrix Completion Algorithm (NN-based+Adam)

### Netflix dataset
https://colab.research.google.com/drive/10wc_ViKCkdIKrYfXoaxzNkpGt_y-oBNq?usp=sharing


### Expedia dataset
https://colab.research.google.com/drive/1JG8rfqfAN9tAbaT9fhMazpSH6vbZX72Z?usp=sharing

<!-- Final submission commit：Ready -->
