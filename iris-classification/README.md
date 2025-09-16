# Iris Classification

This project analyzes the famous Iris dataset using Python.  
It includes data exploration, visualization, and machine learning models for classification.

## Dataset
- **Source**: Built-in dataset from `scikit-learn`.
- **Samples**: 150 flower records.
- **Features**: Sepal length, sepal width, petal length, petal width.
- **Target**: 3 species (Setosa, Versicolor, Virginica).

## Steps
1. **Exploratory Data Analysis (EDA)**  
   - Summary statistics  
   - Pair plots and distributions  

2. **Feature Processing**  
   - Standardization of features  

3. **Modeling**  
   - Logistic Regression  
   - K-Nearest Neighbors (KNN)  
   - Decision Tree  

4. **Evaluation**  
   - Accuracy score  
   - Classification report  
   - Comparison of models  

## Results
- All models achieved accuracy above **90%** on the test set.  
- Petal length and petal width were the most important features for classification.  

## Requirements
- Python 3.x  
- pandas, numpy, seaborn, matplotlib  
- scikit-learn  

Install dependencies with:
```bash
pip install -r requirements.txt
