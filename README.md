# Salary Prediction Regression with Deep Learning & Optuna

## 📌 Project Overview

This repository contains a comprehensive Jupyter Notebook (`Salary Prediction Regression with Deep Learning.ipynb`) focused on predicting employee salary rates using an artificial neural network (ANN) approach. The objective is to estimate numerical individual earnings (`Salary`) by mapping dynamic linear interactions and complex multi-layer configurations derived from baseline workforce indicators. These features include age matrices, genders, standardized education categories, job roles, and specific career history landmarks.

By deploying **Optuna**, an automated hyperparameter tuning suite, the workflow explores wide combinations of multi-layered architectural shapes, backpropagation step variations, and gradient tracking modes. The pipeline evaluates deep sequential configurations built via Keras/TensorFlow to identify structural options that minimize linear variances and root-error metrics.

---

## 🛠️ Project Workflow

The implementation builds a fully integrated regression pipeline divided into four structured architectural phases:

1. **Exploratory Data Analysis (EDA) & Summary Statistics**:
* Loading the tabular workforce structure (`Salary_Data.csv`) containing demographic features and career details.
* Tracking dataset structural properties, observation totals, and structural counts via descriptive pandas validations such as `data.describe(include='all')` and `value_counts()` mappings.


2. **Feature Engineering & Pre-processing**:
* **Categorical Feature Encoding**: Grouping overlapping category strings (e.g., standardizing text variants across `Education Level` metrics) and converting discrete parameters into numeric multi-column arrays using one-hot representation patterns.
* **Feature Normalization**: Implementing the scikit-learn `StandardScaler` to handle multi-scale raw numeric matrices (such as `Age` bounds and `Years of Experience` vectors). This transforms the data into uniform distributions, protecting the deep layers against early gradient saturation or scaling biases.


3. **Hyperparameter Tuning Environment (Optuna)**:
* Constructing an optimized parameter objective function focused on minimizing Mean Squared Error (MSE) constraints over validation sets.
* Orchestrating an automated optimization loop evaluating a range of structural combinations across several criteria:
* **Optimizers**: `SGD`, `Adam`, `RMSprop`, `Adagrad`, `Adadelta`, `Nadam`.
* **Learning Rates**: Dynamic log-scale adjustments guiding the step size of backpropagation.
* **Hidden Layer Widths**: Sizing individual neuron depths across successive hidden `Dense` layers.
* **Batch Training Scales**: Testing different batch sizes and epoch caps to optimize training speed and generalization performance.




4. **Model Finalization & Performance Evaluation**:
* Rebuilding and training the optimized Keras `Sequential` network configuration using the top parameters selected from the tuning objective trials.
* Validating model prediction performance against unseen test data distributions using comprehensive regression metrics, including the $R^2$ Coefficient of Determination ($R^2\_score$), Mean Absolute Error ($MAE$), and Root Mean Squared Error ($RMSE$).



---

## ⚙️ Model Limitations & Operational Constraints

* **Task-Related Thresholds**: Specific parameters applied across the pipeline—such as outlier boundaries, data grouping labels, structural parameters within the automated search objective, and standard learning filters—are **task-related thresholds**. Rather than aligning with general-purpose out-of-the-box machine learning defaults, they were configured to address the specific distributions and scale characteristics of this workforce income dataset.
* **Computational & Hardware Constraints**: Because the deep neural network transformations, iterative numerical backpropagation loops, and automated Optuna optimization cycles were executed on a **local machine**, model performance metrics and training stability trends may reflect "not ideal" configurations. Limited local computational resources constrained the ability to use wider validation folds or complete an exhaustive, higher-dimensional search over all parameter spaces.

---

## 💻 Technologies Used

* **Python 3.x**
* **Data Manipulation & Processing**: Pandas, NumPy
* **Data Visualization**: Matplotlib, Seaborn
* **Deep Learning Frameworks**: TensorFlow, Keras (`Sequential`, `Dense`)
* **Hyperparameter Optimization Suite**: Optuna
* **Statistical Performance Evaluation**: Scikit-Learn (`StandardScaler`, `train_test_split`, `r2_score`, `mean_absolute_error`, `mean_squared_error`)

---

## 🚀 How to Use

1. **Dataset**: Place your source tabular file `Salary_Data.csv` inside the same workspace or working directory containing your notebook project.
2. **Dependencies**: Set up your workspace and install the required library package distributions:
```bash
pip install pandas numpy seaborn matplotlib scikit-learn tensorflow optuna

```


3. **Execution**: Open the `Salary Prediction Regression with Deep Learning.ipynb` notebook file inside your Jupyter ecosystem and execute the cells sequentially. This allows you to track data parsing, view the hyperparameter optimization steps, and inspect the final regression error rates and performance metrics.
