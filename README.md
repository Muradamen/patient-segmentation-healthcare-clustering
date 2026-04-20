
# 🏥 Patient Segmentation in Healthcare using Gower Distance & Hierarchical Clustering

## 📌 Project Overview
This project develops a **real-world patient segmentation system** using **unsupervised machine learning** techniques designed for **mixed-type healthcare data**.

The solution leverages **Gower Distance** and **Hierarchical Clustering** to group patients based on demographic, clinical, utilization, and financial attributes—enabling actionable healthcare insights.

---

## 🎯 Business Problem
Healthcare systems generate large volumes of heterogeneous data, but often lack effective segmentation strategies.

Without segmentation:
- High-risk patients go unidentified
- Resources are inefficiently allocated
- Personalized care is limited

👉 This project addresses these challenges by discovering **hidden patient segments** to support **data-driven decision-making**.

---

## 📊 Dataset Summary
- **Total Records:** 2,000 patients  
- **Total Features:** 16  

### Feature Categories:
- **Demographics:** Age, Gender, State, City  
- **Clinical:** BMI, Number of Chronic Conditions, Primary Condition  
- **Utilization:** Annual Visits, Days Since Last Visit  
- **Financial:** Average Billing Amount, Insurance Type  

---

## ⚠️ Key Challenge: Mixed Data Types

Healthcare datasets typically include:

- **Numerical features:** Age, BMI, Billing Amount  
- **Categorical features:** Gender, Insurance Type, Condition  

### 🚫 Why Standard Methods Fail
Traditional clustering methods like **K-Means**:
- Require purely numeric input  
- Depend on Euclidean distance  
- Are distorted by one-hot encoding  

---

## ✅ Solution Approach

### 🔹 1. Data Preprocessing
- Missing value imputation (median/mode)
- Removal of non-informative features
- Data consistency checks

### 🔹 2. Distance Metric
- Applied **Gower Distance** to properly measure similarity across mixed data types

### 🔹 3. Clustering Algorithm
- Used **Agglomerative Hierarchical Clustering**
- Linkage method: Average

### 🔹 4. Model Evaluation
- Silhouette Score (precomputed distance matrix)
- Dendrogram analysis

### 🔹 5. Cluster Interpretation
- Statistical profiling
- Business-oriented segmentation

---

## 🧠 Methodology Pipeline

1. Data Loading & Inspection  
2. Data Cleaning & Preparation  
3. Gower Distance Matrix Computation  
4. Hierarchical Clustering  
5. Cluster Evaluation  
6. Insight Generation  

---

## 📈 Key Results & Insights

The model identifies distinct patient segments such as:

- **High-Cost Chronic Patients** → Require continuous monitoring  
- **Frequent Service Users** → High utilization patterns  
- **Low-Risk Young Patients** → Preventive care opportunities  
- **Inactive Patients** → Re-engagement strategies  

These segments can support:
- Risk stratification  
- Cost optimization  
- Targeted interventions  

---

## 🧪 Tech Stack

| Category        | Tools |
|----------------|------|
| Language        | Python |
| Data Handling   | Pandas, NumPy |
| Visualization   | Matplotlib, Seaborn, Plotly |
| ML Algorithms   | Scikit-learn, Scipy |
| Distance Metric | Gower |
| Deployment      | Streamlit |

---

## 🚀 Interactive Application (Streamlit)

This project includes a **Streamlit web app** that allows users to:

- Upload custom datasets  
- Perform clustering dynamically  
- Visualize cluster distributions  
- Explore patient segments  

### ▶ Run Locally
```bash
streamlit run app/app.py
````

---

## 🌍 Live Demo

*(Add your deployed Streamlit link here)*

---

## 📂 Project Structure

```
patient-segmentation-healthcare-clustering/
│  patient_segmentation_analysis.ipynb
│app.py
│
├── reports/
│   └── figures/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📊 Evaluation Metrics

* **Silhouette Score (Precomputed Distance)**
* Cluster Separation Analysis
* Dendrogram Visualization

---

## 💡 Key Learnings

* Importance of selecting appropriate distance metrics
* Handling real-world mixed-type data
* Limitations of traditional clustering methods
* Translating machine learning results into business insights

---

## 🔥 Why This Project Stands Out

✔ Real-world healthcare application
✔ Handles mixed data (advanced ML concept)
✔ Uses appropriate distance metric (Gower)
✔ Strong business interpretation
✔ Interactive deployment included

---

## 🧩 Future Enhancements

* Cluster explainability using surrogate models
* Automated feature weighting
* Integration with real-time healthcare systems
* Advanced dashboard with KPIs

---

## 👤 Author

**Murad Amin**
Data Analyst | AI Practitioner

---

## 📬 Contact

www.linkedin.com/in/muradamin


