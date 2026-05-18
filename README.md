## Predicting heart disease Prediction using machine learning

i will use a Python-based machine learning and data science libraries in an attempt to build a machine learning model capable of predicting whether or not someone has heart disease based on their medical attributes.

We're going to take the following approach( following the ML Workflow):

Problem definition
Data
Evaluation
Features
Modelling
Experimentation

1. Problem Definition
In a statement,
Given a patient’s clinical measurements, can we accurately predict whether they are likely to have heart disease or not?

2. Data
 it is available on Kaggle. https://www.kaggle.com/datasets/mfarhaannazirkhan/heart-dataset

 This dataset contains 1,888 records merged from five publicly available heart disease datasets. It includes 14 features that are crucial for predicting heart attack and stroke risks, covering both medical and demographic factors. Below is a detailed description of each feature.


3. Evaluation
If we can reach 95% accuracy at predicting whether or not a patient has heart disease during the proof of concept, we'll pursue the project.

4. Features
This is where you'll get different information about each of the features in your data.

Create data dictionary

age - age in years

sex - (1 = male; 0 = female)

cp - chest pain type
0: Typical angina: chest pain related decrease blood supply to the heart
1: Atypical angina: chest pain not related to heart
2: Non-anginal pain: typically esophageal spasms (non heart related)
3: Asymptomatic: chest pain not showing signs of disease

trestbps - resting blood pressure (in mm Hg on admission to the hospital) anything above 130-140 is typically cause for concern

chol - serum cholestoral in mg/dl


fbs - (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
'>126' mg/dL signals diabetes

restecg - resting electrocardiographic results
 Values: 0 = Normal, 1 = ST-T wave abnormality, 2 = Left ventricular hypertrophy.


thalach - maximum heart rate achieved(Numeric)

exang - exercise induced angina (1 = yes; 0 = no)

oldpeak - ST depression induced by exercise relative to rest looks at stress of heart during excercise unhealthy heart will stress more

slope - the slope of the peak exercise ST segment
0: Upsloping: better heart rate with excercise (uncommon)
1: Flatsloping: minimal change (typical healthy heart)
2: Downslopins: signs of unhealthy heart

ca - number of major vessels (0-3) colored by flourosopy
colored vessel means the doctor can see the blood passing through
the more blood movement the better (no clots)

thal - thalium stress result
1,3: normal Values: 1 = Normal, 2 = Fixed defect, 3 = Reversible defect.


target -Outcome variable (heart attack risk). Values: 1 = more chance of heart attack, 0 = less chance of heart attack.

5.Tools

pandas for data analysis.

NumPy for numerical operations.

Matplotlib/seaborn for plotting or data visualization.

Scikit-Learn for machine learning modelling and evaluation.
