# 📰 FlipItNews - News Article Classification using NLP

## 📌 Project Overview
The **FlipItNews** project focuses on building a **multi-class news classification system** that can automatically categorize news articles into topics such as **Politics, Technology, Sports, Business, and Entertainment** using **Natural Language Processing (NLP)** and **Machine Learning**.

The dataset contains news articles from the company’s internal database.  
The goal is to clean, process, vectorize, and train multiple ML models to find the best-performing classifier.

---

## 🎯 Objective
- Process raw news articles and clean the text.
- Remove noise (stopwords, special characters, punctuation).
- Apply **Tokenization** and **Lemmatization**.
- Transform data using **Bag of Words (BoW)** and **TF-IDF**.
- Train and compare **multiple machine learning models**.
- Evaluate performance using **classification metrics** and **confusion matrix**.

---

## 📂 Dataset
**Attributes:**
- `Article` → Text content of the news.
- `Category` → Target variable (e.g., Politics, Technology, etc.).

---

## 🛠️ Tech Stack
- **Programming Language:** Python 🐍
- **Libraries:**
  - `pandas`, `numpy` → Data handling
  - `matplotlib`, `seaborn` → Visualization
  - `nltk` → NLP preprocessing
  - `scikit-learn` → Machine Learning models & evaluation

---

## 📊 Project Workflow

### 1️⃣ Import Libraries & Load Data
- Load CSV dataset into Pandas DataFrame.
- Explore dataset shape and class distribution.

### 2️⃣ Text Preprocessing
- Remove non-letter characters.
- Convert text to lowercase.
- Remove **stopwords** using NLTK.
- Tokenize text into words.
- Apply **Lemmatization** (NLTK WordNetLemmatizer).

### 3️⃣ Feature Engineering
- Encode target variable using **LabelEncoder**.
- Vectorize text using:
  - **Bag of Words** (`CountVectorizer`)
  - **TF-IDF** (`TfidfVectorizer`)
- Limit feature dimensions to **max_features=5000**.

### 4️⃣ Train-Test Split
- Split data into **75% train** and **25% test**.

### 5️⃣ Model Training
Implemented models:
- **Multinomial Naive Bayes**
- **Decision Tree Classifier**
- **K-Nearest Neighbors (KNN)**
- **Random Forest Classifier**
- **Support Vector Machine (SVM)** (Linear Kernel)

### 6️⃣ Evaluation
- Accuracy score
- Precision, Recall, F1-score
- Confusion matrix

---


