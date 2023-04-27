# Classification-Models
Assignment 2 - Classification Models repository contains project for 36106 Machine Learning Algorithms and Applications

## Business Understanding
The purpose of this project is to build a model that predicts if a customer will repurchase a product or not. To achieve this, I will be using different classification models and and their hyperparameters, a dataset that contains customer information and their purchase history . The goal is to create a model that can accurately predict whether a customer will repurchase a product or not, in order to help businesses identify potential repeat customers and implement strategies to retain them.
## Data Understanding
I analysed the data provided by the company and found that it contains various features such as age, gender, income, credit score, car model, car segment, and previous purchases. The dataset contains 200,000 rows and 11 columns. There are 10,000 observations in the dataset, and the target variable is binary (1 for buy and 0 for not buy). I also found that the dataset has some missing values that need to be imputed.
## Data Preparation
I performed several steps to prepare the data for modeling. I started with data cleaning, where I removed unnecessary features and imputed the missing values. For imputation, I used the mean value for continuous features and the mode value for categorical features. I also converted categorical variables into numerical variables using one-hot encoding. Finally, I split the data into training and testing sets with a ratio of 80:20.
## Modeling
I used six different classification models, including Random Forest Classifier, Support Vector Machines (SVM), Decision Tree, KNN, Naive Bayes and Bagging Decision Tree, to create our model. I used the hyperparameter grid for each model to search for the optimal hyperparameters. I then trained the models on the training data and then evaluated the performance of each model using accuracy as the evaluation metric. I have also tested the model using precision, recall, and F1-score
## Evaluation
I found that BaggingClassifier( Decision tree), Random Forest Classifier, Decision Tree and SVM performed well compared to KNN and Naive Bayes.
BaggingClassifier (Decision Tree) had the highest accuracy of 0.992 followed by Random Forest Classifier with accuracy of 0.99 and F1-score of 0.84, and Decision Tree with accuracy of 0.98. I also observed that some models had a high false-negative rate, indicating that they were unable to correctly predict the customers who would buy.
## Deployment
I recommend further experimentation with the Bagging Decision Tree Classifier model as it had the highest accuracy and F1-score. The next steps could include feature selection and engineering to improve the performance of the model. I also recommend exploring other classification algorithms such as Gradient Boosting Classifier, XGBoost, and Neural Networks to improve the performance of the model.
## Issues:
During the experiment, several issues were faced, including:
1. Data Cleaning: One of the initial challenges is to clean the data and remove or handle missing values, outliers, and errors. This was solved by removing missing values, imputing them with mean, median or mode, and removing or correcting outliers.
2. Imbalanced dataset: The dataset used in the experiment was imbalanced, with a majority of the observations belonging to one class. This can lead to biased models that predict the majority class more often, which can be detrimental to the business objective. One solution to this is to use techniques such as oversampling, under sampling, or generating synthetic samples to balance the dataset.
3. Missing values: The dataset contained missing values, which were dropped in the code provided. However, a better solution is to impute missing values using techniques such as mean imputation, median imputation, or regression imputation.
4. Limited hyper-parameter tuning: Although hyper parameter tuning was performed in the experiment, the range of hyperparameters tested was limited. A broader search over a wider range of hyperparameters could potentially lead to better model performance.
5. Model Evaluation: It can be challenging to evaluate the performance of the model, especially when dealing with imbalanced data or multiple evaluation metrics. This was addressed by using appropriate evaluation metrics such as accuracy, precision, recall, F1-score, AUC-ROC, and confusion matrix.
## Recommendations:
1. Collect additional data: Collecting more data may help improve the accuracy of the model, particularly if there are features that are not included in the current dataset that may be predictive of whether a customer will repurchase a product or not.
2. Feature engineering: I can explore creating new features from the existing ones, such as combining certain features or creating interaction terms.
3. Model ensemble: I can explore ensembling multiple models to create a more robust and accurate prediction model.
4. Model deployment: If the model achieved the required outcome for the business, I can recommend deploying this solution into production. This may require additional steps such as testing the model on new data, integrating the model with existing systems, and implementing a feedback loop to continuously improve the model's performance.
## Results:
The results of the experiment show that the Bagging Decision Tree achieved highest accuracy score of 99.2%, followed by Random Forest Classifier (99%), SVM (98.2%), Decision Tree (98%), KNN (95%), and Naive Bayes (78.0%).
The Bagging Decision Tree Classifier is a good candidate for the final model, as it achieved the highest accuracy score.
## Conclusion:
In this project, I used the CRISP-DM methodology to build six different classification models based on different methodology that predicts if a customer will buy or not.    
I tested six different classification models and found that Bagging Decision Tree Classifier had the best performance. I recommend further experimentation to improve the performance of the model by feature selection and engineering, and exploring other classification algorithms. This model can be used by the company to target potential buyers and increase sales.
