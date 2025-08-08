# 🚀 Machine Learning & Deep Learning Learning Repository

Welcome to my comprehensive Machine Learning and Deep Learning learning journey! This repository contains hands-on implementations, experiments, and demonstrations of various ML/DL concepts using Python.

## 📊 Repository Structure

```
ML_DL_Learning/
├── 📓 Notebooks/
│   ├── simple-linear-regession.ipynb    # Linear regression implementation
│   ├── tensor-demo.ipynb                # Tensor operations with NumPy
│   ├── working-with-csv.ipynb           # CSV data manipulation
│   ├── Dataset-import-export.ipynb      # Kaggle dataset operations
│   ├── mnist_classification.ipynb       # Deep learning MNIST digit classification
│   └── admission_perdication.ipynb      # Neural network admission prediction
├── 📂 data/
│   └── placement.csv                    # Student placement dataset
├── 🔧 Configuration/
│   └── packages.txt                     # Required Python packages
└── .gitignore                          # Git ignore rules
```

## 🎯 What's Inside

### 📈 Simple Linear Regression

**File:** [`simple-linear-regession.ipynb`](simple-linear-regession.ipynb)

A complete implementation of linear regression to predict student placement packages based on CGPA:

- 📊 Data visualization using matplotlib with scatter plots
- 🔄 Train-test split implementation (80/20 split)
- 🤖 Linear regression model training with scikit-learn
- 📉 Model evaluation and performance metrics (R² score: ~0.78)
- 🎨 Beautiful scatter plots with regression line visualization

**Model Performance:** Achieves ~78% accuracy (R² score: 0.78) in predicting placement packages.

### 🧠 MNIST Digit Classification

**File:** [`mnist_classification.ipynb`](mnist_classification.ipynb)

Deep learning implementation for handwritten digit recognition using TensorFlow/Keras:

- 🖼️ MNIST dataset loading and preprocessing (28x28 grayscale images)
- 📊 Data normalization (pixel values scaled to 0-1 range)
- 🏗️ Multi-layer neural network architecture:
  - Flatten layer for input (784 features)
  - Dense layers: 128 → 64 → 32 neurons with ReLU activation
  - Output layer: 10 neurons with softmax for digit classification
- 📈 Model training with 20 epochs and validation split
- 🎯 Performance visualization with accuracy and loss plots
- 📊 Model evaluation using scikit-learn metrics

**Model Architecture:** Sequential neural network with 4 hidden layers
**Dataset:** 60,000 training images, 10,000 test images

### 🎓 Graduate Admission Prediction

**File:** [`admission_perdication.ipynb`](admission_perdication.ipynb)

Neural network regression model to predict graduate admission chances:

- 📥 Kaggle dataset integration using kagglehub
- 🧹 Data preprocessing and feature engineering
- 📊 Feature scaling using MinMaxScaler
- 🏗️ Simple neural network architecture:
  - Input layer: 7 features (GRE, TOEFL, University Rating, etc.)
  - Hidden layer: 7 neurons with ReLU activation
  - Output layer: 1 neuron with linear activation for regression
- 📈 Model training with validation monitoring
- 📊 R² score evaluation for regression performance
- 📉 Loss and accuracy visualization

**Features:** GRE Score, TOEFL Score, University Rating, SOP, LOR, CGPA, Research Experience

### 🧮 Tensor Operations Demo

**File:** [`tensor-demo.ipynb`](tensor-demo.ipynb)

Fundamental tensor operations using NumPy for understanding ML/DL foundations:

- 📐 Understanding array dimensions with `.ndim`
- 🔢 Working with scalars, vectors, and matrices
- 🛠️ Basic tensor manipulations and operations

### 📄 CSV Data Handling

**File:** [`working-with-csv.ipynb`](working-with-csv.ipynb)

Essential data manipulation techniques for ML preprocessing:

- 📥 Reading CSV files with pandas
- 🏷️ Custom column naming and data structure
- 📋 Data exploration and preprocessing techniques

### 🌐 Dataset Management

**File:** [`Dataset-import-export.ipynb`](Dataset-import-export.ipynb)

Professional dataset management with Kaggle API integration:

- 🔐 Kaggle API authentication setup
- ⬇️ Automated dataset downloading and extraction
- 🗂️ Dataset cleanup and organization
- 📊 Face mask detection dataset example

## 📦 Required Packages

Install all dependencies listed in [`packages.txt`](packages.txt):

```txt
numpy          # Numerical computing and array operations
pandas         # Data manipulation and analysis
matplotlib     # Data visualization and plotting
scikit-learn   # Machine learning algorithms and tools
tensorflow     # Deep learning framework
keras          # High-level neural networks API
kagglehub      # Kaggle dataset integration
```

## 🚀 Getting Started

### Quick Setup

1. **Clone the repository:**

```bash
git clone <your-repository-url>
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

4. **Set up Kaggle API (optional):**

   - Download `kaggle.json` from your Kaggle account
   - Place it in the project root directory (not tracked in git)

5. **Launch Jupyter Notebook:**

```bash
jupyter notebook
```

### Running the Deep Learning Demos

1. **MNIST Classification:**

   - Open [`mnist_classification.ipynb`](mnist_classification.ipynb)
   - Run cells sequentially to train the digit classifier
   - View digit images and model performance plots

2. **Admission Prediction:**
   - Open [`admission_perdication.ipynb`](admission_perdication.ipynb)
   - Execute cells to train the regression model
   - Analyze R² score and prediction accuracy

## 📊 Dataset Information

### Student Placement Dataset

**Location:** [`data/placement.csv`](data/placement.csv)

**Features:**

- **CGPA**: Student's Cumulative Grade Point Average (range: ~4.5-8.6)
- **Package**: Placement package amount in lakhs (target variable, range: ~1.5-4.0)

**Dataset Stats:**

- Total samples: 200 student records
- Perfect for regression analysis and predictive modeling
- Clean data with no missing values
- Strong positive correlation between CGPA and placement package

### MNIST Handwritten Digits

**Source:** Built-in TensorFlow/Keras dataset

**Features:**

- **Images**: 28x28 grayscale images of handwritten digits (0-9)
- **Labels**: Integer labels from 0 to 9
- **Training Set**: 60,000 images
- **Test Set**: 10,000 images

### Graduate Admissions Dataset

**Source:** Kaggle dataset via kagglehub

**Features:**

- **GRE Score**: Graduate Record Examination score
- **TOEFL Score**: Test of English as a Foreign Language score
- **University Rating**: Rating of university (1-5 scale)
- **SOP**: Statement of Purpose strength (1-5 scale)
- **LOR**: Letter of Recommendation strength (1-5 scale)
- **CGPA**: Cumulative Grade Point Average
- **Research**: Research experience (0 or 1)
- **Target**: Chance of Admit (0-1 probability)

## 🎨 Visualizations

The repository showcases various data visualization techniques:

- 📊 **Scatter plots** for data distribution analysis
- 📈 **Regression lines** overlaying actual data points
- 🖼️ **Image visualization** for MNIST digit samples
- 📉 **Training history plots** for loss and accuracy monitoring
- 🎯 **Model performance comparisons** with actual vs predicted values

## 🔧 Configuration

### File Structure

- **[`.gitignore`](.gitignore)**: Excludes sensitive files, virtual environments, and temporary files
- **[`packages.txt`](packages.txt)**: Python package dependencies
- **[`README.md`](README.md)**: This comprehensive documentation

### Security Notes

- Kaggle API credentials (`kaggle.json`) should be excluded from version control
- Virtual environment (`.venv/`) is also excluded from git tracking
- Downloaded datasets may be excluded depending on size and sensitivity

## 📚 Learning Objectives

This repository demonstrates proficiency in:

- ✅ **Linear regression** implementation from scratch
- ✅ **Deep learning** with TensorFlow and Keras
- ✅ **Neural network architectures** for classification and regression
- ✅ **Data preprocessing** and exploratory data analysis
- ✅ **Model training and evaluation** with various metrics
- ✅ **Data visualization** with matplotlib
- ✅ **Working with real-world datasets** and API integration
- ✅ **Kaggle integration** with kagglehub
- ✅ **Best practices** for ML/DL project structure and documentation
- ✅ **Version control** with proper .gitignore configuration

## 🔍 Code Quality Features

- **Consistent naming conventions** across all notebooks
- **Comprehensive documentation** with markdown cells
- **Modular code structure** for easy understanding
- **Professional visualization** with proper labels and colors
- **Error handling** and data validation
- **Reproducible results** with fixed random states
- **Proper data scaling** and preprocessing

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork the repository**
2. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes** with proper documentation
4. **Add tests** if applicable
5. **Submit a pull request** with a clear description

### Contribution Ideas

- Add more ML algorithms (SVM, Random Forest, etc.)
- Implement CNN models for image classification
- Add more datasets and analysis
- Improve visualizations
- Add unit tests

## 📈 Future Enhancements

### Planned Implementations

- 🧠 **Convolutional Neural Networks**: CNN implementations for image tasks
- 📊 **Advanced Datasets**: Multi-class classification, time series data
- 🔍 **Feature Engineering**: Advanced preprocessing techniques
- 🎯 **Model Comparison**: Comprehensive algorithm benchmarking
- 📱 **Interactive Dashboards**: Streamlit/Plotly implementations
- 🏗️ **MLOps Pipeline**: Model deployment and monitoring
- 📈 **Advanced Metrics**: Precision, recall, F1-score analysis

### Technical Improvements

- Add automated testing with pytest
- Implement CI/CD pipeline
- Add Docker containerization
- Create API endpoints for models
- Add model persistence and loading

## 🏆 Learning Outcomes

After exploring this repository, you'll understand:

- How to implement linear regression from scratch
- Deep learning fundamentals with TensorFlow/Keras
- Neural network architecture design
- Data preprocessing and visualization techniques
- Model evaluation and performance metrics
- Working with APIs (Kaggle) for data acquisition
- Best practices for ML/DL project organization
- Version control for data science projects

## 📞 Contact & Support

- **Questions?** Open an issue in the repository
- **Suggestions?** Submit a pull request
- **Collaboration?** Reach out for joint projects

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🎯 Quick Navigation

| Notebook                                                         | Purpose           | Key Concepts                                      |
| ---------------------------------------------------------------- | ----------------- | ------------------------------------------------- |
| [`simple-linear-regession.ipynb`](simple-linear-regession.ipynb) | Linear Regression | Prediction, Visualization, Model Evaluation       |
| [`mnist_classification.ipynb`](mnist_classification.ipynb)       | Deep Learning     | Neural Networks, Image Classification, TensorFlow |
| [`admission_perdication.ipynb`](admission_perdication.ipynb)     | Neural Regression | Deep Learning Regression, Feature Scaling         |
| [`tensor-demo.ipynb`](tensor-demo.ipynb)                         | NumPy Basics      | Array Operations, Dimensions                      |
| [`working-with-csv.ipynb`](working-with-csv.ipynb)               | Data Handling     | CSV Operations, Pandas                            |
| [`Dataset-import-export.ipynb`](Dataset-import-export.ipynb)     | Data Acquisition  | Kaggle API, Automation                            |

---

⭐ **Star this repository if you find it helpful for your ML/DL journey!** ⭐

_Happy Learning! 🚀 Let's build the future with Machine Learning and Deep Learning!_

---

**Last Updated:** December 2024 | **Python Version:** 3.12+ | **Status:** Active Development
