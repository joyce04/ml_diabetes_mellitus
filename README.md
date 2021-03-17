# 

This Repository includes all codes and experimental results for the CS260 group project.

### Project Goal : Detecting Diabetes Mellitus in ICU patients

#### Detecting Diabetes Mellitus in ICU patients
Diabetes mellitus is a chronic health condition that affects more than 34 million people in the United States and is also the 7th leading cause of death nationally \cite{ndsr2020}. Many pieces of research have confirmed that diabetic patients admitted to an Intensive Care Unit (ICU) have higher risks for complications and disease. Due to their insufficient metabolic control, the rates of severe disease are significantly higher in patients with diabetes \cite{sardu2020impact}. However, the comorbidity of patients in ICU with diabetes has not been studied broadly.
Getting rapid information about a patient’s chronic conditions such as heart disease, injuries, or diabetes has been significantly important during the COVID-19 pandemic. ICUs often do not have verified medical histories for incoming patients when the patient is in distress or unresponsive or when there is a transfer delay of the medical record from another medical provider \cite{widsdatathon2021}. The existence of chronic conditions like diabetes is critical to address when determining a patient’s plan of care in the ICU.

#### WiDS Datathon 2021
The Women in Data Science (WiDS) Datathon is a data-focused hackathon organized as a part of the Women in Data Science Conference at Stanford University. The WiDS Datathon 2021 focuses on patient health, with an emphasis on the chronic condition of diabetes, through data from MIT’s GOSSIS (Global Open Source Severity of Illness Score) initiative. This hackathon finished on March 1, 2021, through the Kaggle competitions with 808 teams participating. Winners were announced at the virtual WiDS Conference on March 8, 2021, via live stream \cite{widsdatathon2021}. The goal of this hackathon is to build a binary classification model to determine a patient admitted to an ICU has been diagnosed with Diabetes Mellitus using the First 24 hours of data from the ICU. We had experimented with various range of machine learning algorithms to classify patients with diabetes in the incomplete dataset. We submitted our final classification results on this hackathon.

Kaggle : https://www.kaggle.com/c/widsdatathon2021



Models :

- logistic regression

- k-nearest neighbors

- Gaussian Naive Bayes

- Decision Tree

- AdaBoost

- Multi-layer Perceptron (MLP)

- XGBoost

  

### Installation

If there is any issue installing with requirements.txt, please install the following packages:

- pip install pandas
- pip install numpy
- pip install sklearn
- pip install matplotlib
- pip install seaborn
- pip install networkx
- pip install fancyimpute
- pip install xgboost
- pip install torch
- pip install torch-geometric
