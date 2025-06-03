# Political Climate and Depression Diagnosis in U.S. Mental Health Facilities (2022)

This project investigates whether the political environment of a U.S. state correlates with the likelihood of a depression diagnosis among individuals receiving treatment in public mental health facilities. Using nationally representative data, we apply both statistical and machine learning methods to explore these associations and uncover key risk factors.

> 🔍 Originally developed as a final project for a graduate-level data science course.

---

## 📊 Project Objective

To assess whether patients living in Republican (red) states were more likely to receive a depression diagnosis compared to those in Democratic (blue) states in 2022 — and to identify additional features associated with depression in red states.

---

## 📁 Files Included

- `Final Project.pdf` – Full report with methodology, figures, and results
- `Logistic_Regression.R` – Code for regression analysis
- `ML_final_version.R` – Final cleaned machine learning model pipeline
- `ML w:o Bipolarflg, and Schizoflg.R` – Alternate model without key comorbid variables
- `Access to Care Ranking 2022 Rank.xlsx` – Reference data used for state-level access to care

---

## 📚 Data Source

- **Dataset:** [MH-CLD 2022 – Mental Health Client-Level Data](https://www.samhsa.gov/data/data-we-collect/mh-cld-mental-health-client-level-data)
- **Publisher:** U.S. Substance Abuse and Mental Health Services Administration (SAMHSA)
- **Scope:** Patients receiving care at State Mental Health Agencies (SMHAs) during 2022

---

## 🔬 Methodology

### 1. Data Cleaning & Feature Engineering
- Converted demographic and categorical variables into dummy/binary features
- Defined states as red or blue based on 2020 U.S. presidential election results
- Filtered and sampled data (200,000 patients) due to large file size

### 2. Modeling Approaches
- **Statistical:** Logistic regression with depression diagnosis as the outcome
- **ML Models:** Random Forest, Naive Bayes, and Gradient Boosting
- **Evaluation Metrics:** AUC, precision-recall curves, accuracy

---

## 📈 Key Findings

- **Living in a red state** was significantly associated with increased odds of receiving a depression diagnosis (OR ≈ 1.10, *p* ≈ 0).
- **Comorbidities**, especially bipolar disorder and schizophrenia, were the most predictive features of depression.
- **Random Forest** achieved the highest performance:
  - AUC: 0.877 (with all features), 0.667 (excluding comorbidities)
  - Accuracy: 83.4% (vs. 79% for logistic regression)

### ➕ Bonus Analysis:
Adding political state to the model increased predictive performance:
- **With state info:** AUC = 0.8536, Accuracy = 0.8005
- **Without state info:** AUC = 0.8439, Accuracy = 0.7917

---

## 🧠 Interpretation

While access to care may vary, the analysis suggests that **political climate may correlate with mental health diagnoses**, even after adjusting for other factors. However, individual-level political data was not available — state-level voting was used as a proxy.

---

## ⚠️ Limitations

- **Population Bias:** Dataset includes only those receiving care through SMHAs
- **Political Proxy:** State residence used as proxy for individual political affiliation
- **Potential Confounders:** Socioeconomic, cultural, or policy differences not fully captured

---

## 💡 Future Work

- Include access-to-care indices as covariates
- Use general population datasets or surveys with individual-level political data
- Examine temporal trends across multiple years

---

## 🧰 Tools Used

- R (tidyverse, caret, randomForest)
- Excel
- SAMHSA MH-CLD Dataset
