# STAT-4830-Group-Project
## Group Members: Shuhan Zhang, Xinyu Zhang, Lexuan Chen

## Summary
This project addresses the problem of predicting consumer hotel preferences based on personalized offer sets and relevant hotel and consumer features, using real-world data from Expedia. Two modeling frameworks were explored: a mixture preference learning model optimized via the Frank-Wolfe algorithm, including neural network extensions, and a matrix completion approach based on low-rank factorization methods. The findings indicate that linear models generally outperform neural networks due to limited data size and inherent linearity in consumer decision-making. Additionally, the matrix completion methods proved robust and effective, particularly when incorporating bias-aware initialization and using the Huber loss function to handle sparsity and noisy data. Overall, the study provides valuable insights into modeling consumer preferences and demonstrates practical methods suitable for predicting choices in large-scale recommendation systems.

## Repository Structure Overview
#### src/
Contains the core implementation files, such as frank_wolfe.py and adam_expedia_completion.py, which define the main classes (Problem_FrankWolfe, etc.) and functions used for training and evaluating the preference prediction model.
#### notebooks/
Includes cleaned Jupyter notebooks used for Frank-Wolfe Algorithm and matrix completion. These notebooks import from src/ and provide step-by-step execution of the data preprocessing, model training, and result visualization. There is a readme file summarizing the function and work of demos.
#### tests/
Includes test files and datasets (e.g., train_28-32_10000.csv, test_28-32_1000.csv) used for evaluating model performance and verifying functionality. 
#### docs/
Stores final presentation slides
#### report.md
The final project report summarizing the problem background, modeling approach, empirical results, and key findings.
#### requirements.txt
Lists the Python dependencies required to reproduce the results in a consistent environment (e.g., torch, numpy, pandas).
#### development_history/
Including the past code, notebooks,tests, reports and slides(including self-critiques), figures and diagrams used in reports before and exploration of LLM.

## Setup Instructions
We highly recommend using Google Colab to view and run our final demo. It handles all required installations automatically—just click “Runtime → Run all”, and the full pipeline will execute without any setup.
If you prefer to run the code locally, follow these steps:
1. Prepare a Jupyter Environment
Make sure Jupyter Notebook is installed and properly configured with a Python 3.9+ kernel.
2. Install Dependencies
Use the provided requirements.txt to install all necessary packages:
pip install -r requirements.txt
3. Download Source Files
Make sure you have the following files locally:
The notebook: notebooks/Demo.ipynb
Source code in src/
Any scripts or test files if you want to run from command line or verify with pytest
4. Set the Python Path (if needed)
Ensure your Python environment can locate the src/ module. If needed, add the project root to your PYTHONPATH or append manually in the notebook.
5. Run the Notebook
Open and execute the notebook:
It will import functions from src/, train the model, and display the results—including convergence plots and ranking tables.

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
