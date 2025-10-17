# Frequent Itemset Data Mining for Retail Transaction Datasets
## CS634: Data mining

## Project Overview
This project implements and compares three popular algorithms for association rule mining on transaction datasets from multiple retailers (Amazon, Apple, Best Buy, Costco, Nike). The goal is to identify frequent itemsets and generate association rules that uncover core purchasing patterns.

The algorithms implemented are:
- **Brute Force Algorithm:** A straightforward method that checks all possible item combinations.
- **Apriori Algorithm:** Optimizes mining by pruning infrequent itemsets early.
- **FP-Growth Algorithm:** Uses a tree structure for efficient frequent pattern mining without candidate generation.

Users can select the retailer dataset, specify minimum support and confidence thresholds, and choose which algorithm(s) to run. The project outputs association rules with their confidence values, along with execution times for performance comparison.

## Project Structure
- Data files (`amazon.csv`, `apple.csv`, `bestbuy.csv`, `costco.csv`, `nike.csv`) contain transaction records.
- `main.py`: The main Python script to run the project via command line.
- `notebook.ipynb`: Jupyter notebook version with interactive code and explanations.
- Dependencies: Pandas, apyori, mlxtend, tabulate.

## Setup Instructions

### Prerequisites
- Python 3.x installed on your system.

### Install Required Python Packages
Create and activate a virtual environment (optional but recommended):

#### On Windows Command Prompt:
```
python -m venv venv
venv\Scripts\activate
```

#### On Mac/Linux Terminal:
```
python3 -m venv venv
source venv/bin/activate
```

Install packages:
```
pip install pandas apyori mlxtend tabulate openpyxl
```

### Running the Project

1. Ensure the dataset CSV files are in the same directory as `main.py`.
2. Run the script:

#### On Windows:
```
python main.py
```

#### On Mac/Linux:
```
python3 main.py
```

3. Follow on-screen prompts to:
   - Choose a dataset (store).
   - Enter minimum support value (e.g., 50 for 50%).
   - Enter minimum confidence value (e.g., 60 for 60%).
   - Select algorithm(s) to run (Brute Force, Apriori, FP-Growth, or All).

4. View the generated association rules and execution time comparisons.

## Notes
- The Apriori algorithm generally performs best on larger datasets.
- Brute Force is simple but becomes very slow with larger data.
- FP-Growth provides a fast alternative using tree-based mining.

## Author
Kenil Pravinbhai Avaiya  
NJIT CS 634 - Data Mining  
October 2025
