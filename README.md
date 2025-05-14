#### **Introduction**

In the financial sector, particularly in lending institutions, assessing the creditworthiness of loan applications is a critical process that directly impacts the profitability and stability of banks. Home equity loans, which allow homeowners to borrow against the equity in their property, carry inherent risks due to the possibility of borrower default. This is the reason why lenders must carefully evaluate the likelihood that the borrower will repay it back or not before approving a loan as this evaluation is crucial because if too many borrowers fail to repay their loans, it can lead to significant financial lossess for the lender. 

The Home Equity dataset (HMEQ) contains detailed information about recent home equity loan applications. It includes data such as the loan amount requested, the applicant's employment history, credit background, and whether they eventually defaulted on the loan or not.

This projects aim to use this dataset to build tools that help banks make smarter lending decisions. By combining data analysis and machine learning techniques with human expertise, we can improve the accuracy of credit assessments and reduce the number of loans that are not repaid.

---

#### **Problem Identification**

One of the biggest challenges in lending is identifying which applicants are likely to repay their loans and which may default. If this is not done correctly, banks can suffer large financial losses due to non-performing loans (NPL), which are loans that are not being paid back as agreed. Currently, banks still rely on manual review by loan officers to decide who gets approved. While experienced reviewers are valuable, this method has some key limitations such as:

- It can be slow and difficult to scale.

- Decisions may vary between different reviewers.

- Human judgement can sometimes be influenced by unconscious biases. 

- Important patterns in the data may be missed due to complexity or volumes of applications. 


These issues can result in poor lending decisions, either approving risky applications or rejecting reliable ones.

---

#### **Project Objectives**

Our main objective is to develop an effective and reliable system that helps banks assess the risk of loan defaults among home equity loan applicants using data-driven methods. These are the objectives and goals that we aim to do:

1. To make analysis on the Home equity loan dataset and identify the most important factors that influence whether a borrower will repay or default.

2. To build and test classification machine learning models to predict clients who are likely to default on their loans.

3. To clean and prepare the dataset for modeling to ensure accurate results.

4. To avoid the risk of misclassification of default loans predicted as non-default as this results in high losses. 

---

#### **Dataset Description**

The Home Equity Loans (HMEQ) dataset contains detailed information on 5,960 recent home equity loan applications . It was collected to help lenders understand and predict which applicants are more likely to default (fail to repay) or repay their loans successfully. This dataset includes both demographic and financial details about each applicant, making it suitable for use in credit risk modeling, loan approval decisions, and predictive analytics.

- **Total Records**: 5,960 recent home equity loan applications.

- **Default Rate**: Approximately 20% (1189 out of 5960 applicants defaulted).

- **Type of Variability**: Loan Financial Varaibles,  Purpose and Employment Variables, and Credit History Variables. In total, there are 12 features and 1 target variable.

- **Target Variable**: `Bad` indicates whether the borrower defaulter (1) or repaid (0). 

<br>

| Variable  | Description                                                                 |
|-----------|-----------------------------------------------------------------------------|
| BAD       | 1 = Client defaulted ; 0 = Loan repaid on time                                     |
| LOAN      | Amount of the laon approved for the home equity loan                                             |
| MORTDUE   | Amount still due on the existing mortgage                                  |
| VALUE     | Current market value of the property                                       |
| REASON    | Purpose of the loan: HomeImp = home improvement, DebtCon = debt consolidation |
| JOB       | Type of job held by applicant                                              |
| YOJ       | Number of years at current job                                                       |
| DEROG     | Number of major derogatory reports (late payments, collections, charge-offs), indicates past credit problems          |
| DELINQ    | Number of delinquent credit lines (a credit line become delinquent when minimum payments are missed for 30-60+ days)                                    |
| CLAGE     | Age of oldest credit line in months (a credit line is a reusable loan that lets you borrow money up to certain limit)                                    |
| NINQ      | Number of recent credit inquiries (Each time a lender pulls credit report)                                       |
| CLNO      | Total number of credit lines currently open (how many accounts the borrower is managing)                               |
| DEBTINC   | Debt-to-income ratio (%), measure the borrower's ability to manage monthly payments         |