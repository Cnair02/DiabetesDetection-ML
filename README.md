# DiabetesDetection-ML

Problem definition: Diabetes prediction from PIMA dataset

Objective: Explore the dataset, data analysis and build models to predict diabetes.

Skills: Data extraction, EDA, visualization, transformation, model building and evaluation.

Tools: Pandas, Scikit-learn, Matplotlib, Seaborn and Numpy

About the Dataset: The objective of the dataset is to be able to ascertain if a person has diabetes or not based on the attributes that is provided. The data collected are all females above 21 years old of Indian PIMA heritage. The following are the attributes of this dataset,
1. Pregnancies: Number of times pregnant 
2. Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
3. BloodPressure: Diastolic blood pressure (mm Hg)
4. SkinThickness: Triceps skin fold thickness (mm)
5. Insulin: 2-Hour serum insulin (mu U/ml)
6. BMI: Body mass index (weight in kg/(height in m)^2)
7. DiabetesPedigreeFunction: Diabetes pedigree function
8. Age: Age (years)
9. Outcome: Class variable (0 or 1) 268 of 768 are 1, the others are 0.

Topics covered:
1. Data Extraction
2. Exploratory Data Analysis
3. Model Building
4. Model Evaluation

Challenges:
1. Gathering high level understanding about factors affecting diabetes.
2. Reading and evaluating columns and correlation plot with the outcome to be determined.
3. Highlighting the best ways to assess the model prediction. What parameters are the most important to this particular use case.

Conclusion:

Looking at the model evaluation by finding the accuracy, precision and recall we can conclude that for this dataset, Decision Tree and Logistic Classifier is a better fit as compared to Random forest and XGBoost. This could possibly be because the dataset is considerably small thereby making simpler models work more efficiently than ensemble methods.

On further analysis between Decision Tree and Logistic Classifier, Decision Tree will be a better choice because, although logistic classifier has better accuracy rate, but when we check the recall score we can see that it is 74% which is 10% more than that of logistic classifier. 

Since in this case of healthcare sector where diabetes detection which lays more importance where missing a positive (or false negative) is more dangerous so high recall is prioritized. Hence we can conclude that Decision Tree will provide the best model for this particular use case.

