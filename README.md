# 🚀 Machine Learning & Deep Learning Learning Repository

Welcome to my comprehensive Machine Learning and Deep Learning learning journey! This repository contains hands-on implementations, experiments, and demonstrations of various ML/DL concepts using Python.

## 📊 Repository Structure

```
ML_DL_Learning/
├── 📓 Notebooks/
│   ├── simple-linear-regession.ipynb    # Linear regression implementation
│   ├── tensor-demo.ipynb                # Tensor operations with NumPy
│   ├── working-with-csv.ipynb           # CSV data manipulation
│   └── Dataset-import-export.ipynb      # Kaggle dataset operations
├── 📂 data/
│   └── placement.csv                    # Student placement dataset
├── 🔧 Configuration/
│   ├── kaggle.json                      # Kaggle API credentials
│   └── packages.txt                     # Required Python packages
└── .gitignore                          # Git ignore rules
```

## 🎯 What's Inside

### 📈 Simple Linear Regression
**File:** [`simple-linear-regession.ipynb`](simple-linear-regession.ipynb)

A complete implementation of linear regression to predict student placement packages based on CGPA:
- 📊 Data visualization using matplotlib
- 🔄 Train-test split implementation
- 🤖 Linear regression model training with scikit-learn
- 📉 Model evaluation and performance metrics
- 🎨 Beautiful scatter plots with regression line visualization

**Key Features:**
```python
# Data visualization
plt.scatter(df['cgpa'], df['package'], color='red')
plt.plot(x_test, model.predict(x_test), color='blue')
```

### 🧮 Tensor Operations Demo
**File:** [`tensor-demo.ipynb`](tensor-demo.ipynb)

Fundamental tensor operations using NumPy:
- 📐 Understanding array dimensions (`.ndim`)
- 🔢 Working with vectors and matrices
- 🛠️ Basic tensor manipulations

### 📄 CSV Data Handling
**File:** [`working-with-csv.ipynb`](working-with-csv.ipynb)

Data manipulation techniques:
- 📥 Reading CSV files with pandas
- 🏷️ Custom column naming
- 📋 Data exploration and preprocessing

### 🌐 Dataset Management
**File:** [`Dataset-import-export.ipynb`](Dataset-import-export.ipynb)

Kaggle dataset integration:
- 🔐 Kaggle API authentication
- ⬇️ Automated dataset downloading
- 🗂️ Dataset extraction and cleanup
- 📊 Face mask detection dataset example

## 📦 Required Packages

The project uses the following Python libraries (see [`packages.txt`](packages.txt)):

```txt
numpy          # Numerical computing
pandas         # Data manipulation
matplotlib     # Data visualization  
scikit-learn   # Machine learning algorithms
```

## 🚀 Getting Started

1. **Clone the repository:**
```bash
git clone <repository-url>
cd ML_DL_Learning
```

2. **Create a virtual environment:**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r packages.txt
```

4. **Launch Jupyter Notebook:**
```bash
jupyter notebook
```

## 📊 Dataset Information

### Student Placement Dataset
**Location:** [`data/placement.csv`](data/placement.csv)

Contains student placement data with the following features:
- **CGPA**: Student's Cumulative Grade Point Average
- **Package**: Placement package amount (target variable)

Perfect for regression analysis and predictive modeling!

## 🎨 Visualizations

The repository includes various data visualizations:
- 📊 Scatter plots for data distribution
- 📈 Regression line overlays
- 🎯 Model prediction comparisons
- 📉 Performance metrics visualization

## 🔧 Configuration

- **`.gitignore`**: Excludes sensitive files (CSV data, virtual environments, Kaggle credentials)
- **`kaggle.json`**: Kaggle API credentials for dataset access (not tracked in git)

## 📚 Learning Objectives

This repository covers:
- ✅ Linear regression implementation from scratch
- ✅ Data preprocessing and visualization
- ✅ Model training and evaluation
- ✅ Working with real-world datasets
- ✅ Kaggle dataset integration
- ✅ Best practices for ML project structure

## 🤝 Contributing

Feel free to contribute by:
1. 🍴 Forking the repository
2. 🌟 Creating feature branches
3. 📝 Adding new ML/DL implementations
4. 🔄 Submitting pull requests

## 📈 Future Enhancements

Planned additions:
- 🧠 Deep learning implementations
- 📊 More complex datasets
- 🔍 Advanced visualization techniques
- 🎯 Model comparison frameworks
- 📱 Interactive dashboards

## 📞 Contact

Questions or suggestions? Feel free to reach out!

---

⭐ **Star this repository if you find it helpful for your ML/DL journey!** ⭐

*Happy Learning! 🚀*
