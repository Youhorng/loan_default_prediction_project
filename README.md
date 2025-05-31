# Loan Default Prediction Project

![Loan Default Predictor](https://img.shields.io/badge/Loan%20Default-Prediction-blue)

A machine learning project to predict the likelihood of loan default using financial, employment, and loan purpose data. This project includes a Streamlit-based web application for user-friendly interaction.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Problem Identification](#problem-identification)
3. [Project Objectives](#project-objectives)
4. [Dataset Description](#dataset-description)
5. [Features](#features)
6. [Installation](#installation)
7. [Usage](#usage)
8. [Project Structure](#project-structure)
9. [Contributing](#contributing)
10. [License](#license)

---

## **Introduction**

In the financial sector, particularly in lending institutions, assessing the creditworthiness of loan applications is a critical process that directly impacts the profitability and stability of banks. Home equity loans, which allow homeowners to borrow against the equity in their property, carry inherent risks due to the possibility of borrower default. 

Lenders must carefully evaluate the likelihood that a borrower will repay their loan before approving it. This evaluation is crucial because if too many borrowers fail to repay their loans, it can lead to significant financial losses for the lender. Moreover, inaccurate assessments can result in either approving high-risk applicants or rejecting creditworthy ones, both of which can harm the institution's financial health and reputation.

This project aims to address these challenges by leveraging data-driven approaches to improve the accuracy and efficiency of credit risk assessments. By analyzing historical loan data and applying machine learning techniques, we can uncover patterns and insights that help predict borrower behavior more effectively. This not only reduces the risk of defaults but also ensures fairer and more consistent lending decisions.

---

## **Problem Identification**

One of the biggest challenges in lending is identifying which applicants are likely to repay their loans and which may default. If this is not done correctly, banks can suffer large financial losses due to non-performing loans (NPL), which are loans that are not being paid back as agreed. 

### Key Challenges:
- It can be slow and difficult to scale.
- Decisions may vary between different reviewers.
- Human judgment can sometimes be influenced by unconscious biases.
- Important patterns in the data may be missed due to complexity or volumes of applications.

These issues can result in poor lending decisions, either approving risky applications or rejecting reliable ones.

---

## **Project Objectives**

The main objective of this project is to develop an effective and reliable system that helps banks assess the risk of loan defaults among home equity loan applicants using data-driven methods. Specific goals include:

1. **Data Analysis**: Analyze the Home Equity Loan dataset to identify the most important factors influencing loan repayment or default.
2. **Model Development**: Build and test classification machine learning models to predict clients who are likely to default on their loans.
3. **Data Preparation**: Clean and preprocess the dataset to ensure accurate and reliable results.
4. **Risk Mitigation**: Minimize the risk of misclassification, especially predicting default loans as non-default, which can result in significant financial losses.

---

## **Dataset Description**

The **Home Equity Loans (HMEQ)** dataset contains detailed information on 5,960 recent home equity loan applications. It was collected to help lenders understand and predict which applicants are more likely to default (fail to repay) or repay their loans successfully. This dataset includes both demographic and financial details about each applicant, making it suitable for use in credit risk modeling, loan approval decisions, and predictive analytics.

### **Key Details**
- **Total Records**: 5,960 recent home equity loan applications.
- **Default Rate**: Approximately 20% (1,189 out of 5,960 applicants defaulted).
- **Type of Variables**: Loan Financial Variables, Purpose and Employment Variables, and Credit History Variables.
- **Target Variable**: `BAD` indicates whether the borrower defaulted (`1`) or repaid (`0`).

### **Features**
| Variable  | Description                                                                 |
|-----------|-----------------------------------------------------------------------------|
| `BAD`     | 1 = Client defaulted; 0 = Loan repaid on time                               |
| `LOAN`    | Amount of the loan approved for the home equity loan                        |
| `MORTDUE` | Amount still due on the existing mortgage                                   |
| `VALUE`   | Current market value of the property                                        |
| `REASON`  | Purpose of the loan: `HomeImp` = home improvement, `DebtCon` = debt consolidation |
| `JOB`     | Type of job held by the applicant                                           |
| `YOJ`     | Number of years at the current job                                          |
| `DEROG`   | Number of major derogatory reports (e.g., late payments, charge-offs)       |
| `DELINQ`  | Number of delinquent credit lines                                           |
| `CLAGE`   | Age of the oldest credit line in months                                     |
| `NINQ`    | Number of recent credit inquiries                                           |
| `CLNO`    | Total number of credit lines currently open                                 |
| `DEBTINC` | Debt-to-income ratio (%), measures the borrower's ability to manage monthly payments |

---

## **Features**

- **Streamlit Web App**: A user-friendly interface for entering loan application details and viewing predictions.
- **Machine Learning Model**: A trained XGBoost model for accurate predictions.
- **Data Preprocessing**: Includes scripts and processed datasets for feature engineering.
- **One-Hot Encoding**: Handles categorical variables dynamically.
- **Fallback Mechanisms**: Uses median/mode values for missing or invalid inputs.

---

## **Installation**

Follow these steps to set up the project locally:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/loan_default_prediction_project.git
   cd loan_default_prediction_project