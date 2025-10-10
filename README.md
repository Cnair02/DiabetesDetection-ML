# DiabetesDetection-ML

Problem definition: Diabetes prediction from PIMA dataset

Objective: Explore the dataset, data analysis and build models to predict diabetes.

Skills: Data extraction, EDA, visualization, transformation, model building and evaluation.

Tools: Pandas, Scikit-learn, Matplotlib, Seaborn and Numpy

Topics covered:

Data Extraction
Exploratory Data Analysis
Model Building
Model Evaluation

Challenges:

Gathering high level understanding about factors affecting diabetes.
Reading and evaluating columns and correlation plot with the outcome to be determined.
Highlighting the best ways to assess the model prediction. What parameters are the most important to this particular use case.

Conclusion:

Looking at the model evaluation by finding the accuracy, precision and recall we can conclude that for this dataset, Decision Tree and Logistic Classifier is a better fit as compared to Random forest and XGBoost. This could possibly be because the dataset is considerably small thereby making simpler models work more efficiently than ensemble methods.

On further analysis between Decision Tree and Logistic Classifier, Decision Tree will be a better choice because, although logistic classifier has better accuracy rate, but when we check the recall score we can see that it is 74% which is 10% more than that of logistic classifier. 

Since in this case of healthcare sector where diabetes detection which lays more importance where missing a positive (or false negative) is more dangerous so high recall is prioritized. Hence we can conclude that Decision Tree will provide the best model for this particular use case.

