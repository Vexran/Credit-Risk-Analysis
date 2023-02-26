# Credit-Risk-Analysis
## Overview
This analysis aims to understand how to utilize Machine Learning statistical algorithms to make predictions based on data patterns provided. In this challenge, we focus on Supervised Learning using a free dataset from LendingClub, a P2P lending service company, to evaluate and predict credit risk. The reason why this is called "Supervised Learning" is because the data includes a labeled outcome.

To complete this analysis, we use different Machine Learning techniques to train and evaluate the data with unbalanced classes. The dataset from LendingClub has an unstable classification problem due to the number of good loans outweighing the number of risky loans. Therefore, we needed to employ various Machine Learning algorithms to resample the data to balance out the classifications to allow for more meaningful predictions and improve the accuracy score. These algorithms include RandomOverSampler, SMOTE, ClusterCentroids, SMOTEENN, BalancedRandomForestClassifier, and EasyEnsembleClassifier.

## Results
As mentioned in the overview, we use Machine Learning to resample the dataset using Python libraries: sci-kit-learn and imbalanced-learn to evaluate the results and provide a comparison for our analysis.

The original dataset contained 115,675 loan applications in Q1 of 2019. We used the "loan status" to determine whether the application was considered "low" or "high" risk. Applications that had "current" as the "loan status" were classified as "low risk" and the remaining as "high risk." This reduced the dataset to 68,817 applications, with 99% classified as "low risk."
![Low risk image](https://user-images.githubusercontent.com/113754027/221385764-5ae0404e-40d8-48b6-8847-30a3776382ad.png)
Using a method to split the data for training vs. testing, 51,366 "low risk" and 246 "high risk" applications were categorized into the training set. It was not that bad. 
# Delieverable 1
## OverSampling
RandomOverSampler Model randomly selects from the minority class and adds it to the training set until both classifications are equal. The results classified 51,366 records each as High Risk and Low Risk. 

![counter low risk](https://user-images.githubusercontent.com/113754027/221385829-ae72d620-e2ec-45bb-98c2-e23bed92e98b.png)
The balanced accuracy was a 66%
![the balanced accuracy](https://user-images.githubusercontent.com/113754027/221385843-c706b465-d5eb-4985-872e-9c9dd816319d.png)
the high risk showed that the precision was 1% with a recall that was 72% showing the model F1 result of 2%
the low risk precision was a rate of 100% and recall with 60%
![array](https://user-images.githubusercontent.com/113754027/221385940-e291dd6c-763b-490d-b6f3-ed87ca909824.png)
![percentage](https://user-images.githubusercontent.com/113754027/221385943-be09eba7-3dac-4dfa-b0d7-df6170c9c62e.png)
SMOTE (Synthetic Minority Oversampling Technique) Model, like RandomOverSampler, increases the size of the minority class by creating new values based on the value of the closest neighbors to the minority class instead of random selection. 
The accuracy of the balance score improved a bit by 65.1%
![balance score](https://user-images.githubusercontent.com/113754027/221385979-0b57000f-99ae-4e0c-8636-f1c355419688.png)
Like RandomOverSampler, the "High Risk" precision rate again was only 1%, with the recall degraded to 61%, giving this model an F1 score of 2%.
"Low Risk" had a precision rate of 100% and an improved recall at 69%.
![array 2](https://user-images.githubusercontent.com/113754027/221385998-e28e1201-5199-4a0a-8c97-a73c19887ed8.png)
![low risk](https://user-images.githubusercontent.com/113754027/221386001-4a2f8aac-e6dd-48b4-8f43-55a1ca58a918.png)
ClusterCentroids Model is an algorithm that identifies clusters of the majority class to generate synthetic data points representing the groups. The model classified 246 records, each as High Risk and Low Risk. 
![cluster model](https://user-images.githubusercontent.com/113754027/221386095-429ebd03-c7a6-43d1-8e2d-7743c2b1c233.png)
the Balanced accuracy score was lower than the oversampling models at 54.5%.
![balance accuracy](https://user-images.githubusercontent.com/113754027/221386108-e1a11bc7-31cd-4b4f-90ff-a466e75fe44a.png)
The "High Risk" precision rate again was only at 1% with the recall at 69% giving this model an F1 score of 1%.
"Low Risk" had a precision rate of 100% and with a lower recall at 40% compared to the oversampling models. ![array 3](https://user-images.githubusercontent.com/113754027/221386129-872d8ea0-ec49-4deb-b8a5-e0028dd85a5c.png)
![background](https://user-images.githubusercontent.com/113754027/221386144-cce513d5-a364-4ebb-bdbe-611cc96e2117.png)

## Deliverable 2

#combinations and sampling
SMOTEENN (Synthetic Minority Oversampling Technique + Edited NearestNeighbors) Model combines aspects of both oversampling and undersampling. The model classified 68,460 records as High Risk and 62,011 as Low Risk. 

![counter 2](https://user-images.githubusercontent.com/113754027/221386179-8b323134-db6b-4f30-9e5d-5c4d5f3ca994.png)
The balanced accuracy score improved to 64.5% when using a combined sampling model
![counter 3](https://user-images.githubusercontent.com/113754027/221386203-cad0c60b-2449-4f77-9323-a810b4586be9.png)
The "High Risk" precision rate did not improve by only 1%. However, the recall increased to 72%, giving this model an F1 score of 2%.
"Low Risk" still showed a precision rate of 100%, with the recall at 57%.
![array 4](https://user-images.githubusercontent.com/113754027/221386217-db70b8f0-36fd-45c9-a910-08b92e8d0c63.png)
![risk 2](https://user-images.githubusercontent.com/113754027/221386220-1d1c1f36-0624-429c-8e75-0deab5bb3417.png)

## Deliverable 3
Compare two new Machine Learning models that reduce bias to predict credit risk. The models classified 51,366 as High Risk and 246 as Low Risk.
![counter 5](https://user-images.githubusercontent.com/113754027/221386262-0e35d663-36f5-4f82-a338-9cb95717d381.png)
BalancedRandomForestClassifier Model, two trees of the same size and equal size to the minority class are constructed to represent one for the majority class and one for the minority class.

The balanced accuracy score increased to 78.9% for this model. 
![balanced 2](https://user-images.githubusercontent.com/113754027/221386293-60155e8c-c3ec-435b-af6e-764817d6c5a0.png)
The "High-Risk precision rate increased to 3% with the recall at 70%, giving this model an F1 score of 6%.
"Low Risk" still had a precision rate of 100%, with the recall at 87%.
The top feature by importance was "total_rec_prncp" at 7.9% of the total.
![array 5](https://user-images.githubusercontent.com/113754027/221386306-7f34e290-d970-424d-aefc-2d92f81df271.png)
![risk 3](https://user-images.githubusercontent.com/113754027/221386323-e72f8e8a-d921-4145-b3ce-8847eb7d6c11.png)
![risk 4](https://user-images.githubusercontent.com/113754027/221386329-93086d50-78ed-49b4-adf9-c391f00f305d.png)
EasyEnsembleClassifier Model, a set of classifiers where individual decisions are combined to classify new examples.

The balanced accuracy score increased to 93.2% with this model. 
![calculated](https://user-images.githubusercontent.com/113754027/221386340-96396905-086b-4ee5-a6dc-c2c57ceac908.png)
The "High-Risk precision rate increased to 9% with the recall at 92%, giving this model an F1 score of 16%.
"Low Risk" still had a precision rate of 100%, with the recall now at 94%.
![array 6](https://user-images.githubusercontent.com/113754027/221386388-7c7f5db3-77f3-4f19-951a-1921a31b2c28.png)
![risk 5](https://user-images.githubusercontent.com/113754027/221386390-56aedc29-4e03-4ed4-97ea-982a9fb9efe0.png)
# Summary
In reviewing all six models, the EasyEnsembleClassifer model yielded the best results with an accuracy rate of 93.2% and a 9% precision rate when predicting "High-Risk candidates. The sensitivity rate (aka recall) was also the highest at 92% compared to the other models. The result for predicting "Low Risk" was also the highest, with a sensitivity rate of 94% and an F1 score of 97%. Therefore, if a model needed to be recommended for this type of analysis, then this would be a clear choice.

Ranking of models in descending order based on "High Risk" results:

EasyEnsembleClassifer: 93.2% accuracy, 9% precision, 92% recall, and 16% F1 Score
BalancedRandomForestClassifer: 78.9% accuracy, 3% precision, 70% recall, and 6% F1 Score
SMOTE: 65.2% accuracy, 1% precision, 61% recall, and 2% F1 Score
SMOTEENN: 64.5% accuracy, 1% precision, 72% recall, and 2% F1 Score
RandomOverSampler: 64.0% accuracy, 1% precision, 66% recall, and 2% F1 Score
ClusterCentroids: 54.5% accuracy, 1% precision, 69% recall, and 1% F1 Score
A side note that should be considered is that the original dataset had 99% of the applications classified as "Low Risk," with only 1% of the data organized in the "High Risk" category. This may skew the results significantly as there is a risk that the Machine Learning algorithms are creating clusters drawing from too small of a dataset of actual "High Risk" applications. This margin of risk is not something that banks would be comfortable accepting.







