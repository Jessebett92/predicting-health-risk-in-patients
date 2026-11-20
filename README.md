Predicting Health Risk in Patients

A. Project Overview
         My research question for this project is: Can we accurately predict patients’ health risk for early medical intervention? My efforts to answer this question stem from the organizational need to support earlier healthcare intervention to improve patient outcomes. My approach is to develop a machine learning model that can identify individuals more likely to develop chronic illnesses. In a healthcare setting, recognizing high risk patients before their condition worsens will help providers prioritize outreach, screenings, and care plans.
            This project addressed the need for a data-driven decision-support tool that can help healthcare professionals make more informed decisions about which patients need prioritized attention. The solution is intended to complement clinical expertise by providing risk predictions based on patterns in a patient’s lifestyle and health data.
My project includes preparing healthcare data, performing feature engineering, training a machine learning classification model, and evaluating its performance. The project focuses on classifying patients into two high-risk and low-risk groups based on their likelihood of developing chronic disease. The analysis includes data cleaning, preprocessing, feature engineering, model development, validation, and interpretation of results.
            The project does not include clinical diagnosis, treatment planning, or deployment in a live healthcare setting. It is limited to the available Kaggle dataset and the features I chose to evaluate. The goal is to build and evaluate a predictive model that can support healthcare decision-making.
The solution is a supervised machine learning workflow designed to predict a patient's chronic disease risk level. The project uses the Kaggle Healthcare Real-Time Risk Dataset. After preparing the data, I trained an XGBoost classification model.
            I used Python for this project along with data analysis libraries such as NumPy, Pandas, Matplotlib, Seaborn, and Scikit-learn. I used these tools for data cleaning, exploratory analysis, feature engineering, model training, and evaluating the solution. My methodology included supervised learning, train-test splitting, cross-validation, and performance evaluation using metrics such as accuracy, precision, recall, F1-score, and ROC AUC. I also created visualizations such as a confusion matrix and ROC curve to support interpretation of the solution.
B. Project Execution
         My project plan included selecting a healthcare dataset, cleaning and preparing the data, identifying my target y variable, engineering relevant features, training machine learning models, evaluating model performance, and creating visuals. The plan included implementing cross-validation and a test dataset to verify the model’s generalizability and ensure reliable results.
            The project followed this structure closely. However, I made some adjustments during development based on the characteristics of the dataset I used. For example, I prioritized the recall score that my model achieved. This is because correctly identifying high-risk individuals matters far more in healthcare than achieving high overall accuracy.
The project planning methodology followed a structured data science workflow. This included defining the problem, obtaining, preparing, and analyzing the data, evaluating the model, and communicating the results to stakeholders.
The project timeline included milestones for key steps such as dataset selection, data cleaning, feature engineering, model training, evaluation, and reporting. Some milestones took longer than expected because of data preparation and model evaluation. These delays were manageable because the workflow was flexible enough to refine these project steps.
Data review and model tuning took up most of my project time because these steps required careful checking to ensure the dataset was usable and the model results were reliable. Even so, I completed the project on schedule.
C. Data Selection and Collection
         The plan for this project was to use a publicly available healthcare dataset that resembled real-time health data. That is why I chose the Kaggle dataset called “Healthcare Real-Time Risk Dataset.” One difference was that I wasn’t prepared for how complex a multi-feature target column would be when training a machine learning model. I needed to review the project goals and determine how to achieve them, so I prepared the dataset for binary classification of the target feature.
One obstacle I encountered during data collection was ensuring the dataset had the right structure and that I chose the correct target variable for the project. Because the data needed to support a classification problem, the target column needed to clearly represent patient risk categories. After removing the multi-feature characteristic from my target column, the dataset also needed to be reviewed for missing values, inconsistent formatting, and potential class imbalance.
There were no unplanned data governance issues because the dataset was publicly available and did not resemble real patient data. However, since the dataset still involved healthcare-related information, it was still necessary to handle it responsibly. The project did not attempt to identify individuals or combine data with other sources in ways that would raise privacy concerns. It was used only for analysis and stored securely during project development. The Kaggle source was cited appropriately, and the dataset was treated as sensitive because this project aims to handle real healthcare data one day.
C1. Advantages and Limitations of the Dataset
         The dataset had several advantages. It was publicly available, well-structured for analysis, and resembled real-world healthcare data. The provided healthcare and lifestyle features reasonably supported a health-risk prediction model. Because the data was already organized in CSV format, it was well suited for model development in Python.
            There were some limitations. It did not represent real patients. So, there is no way of knowing how often a high-risk case would appear in an actual healthcare setting. I am also not certain if the patterns in the data resemble the full complexity of real clinical environments. After converting the dataset's target column to binary classification, the dataset became imbalanced. I considered these limitations when interpreting the results.
D. Data Extraction and Data Preparation
         I obtained the dataset for this project from Kaggle. I downloaded it as a CSV file named “healthcare_real_time_dataset.” I imported it using Python’s Pandas library.      
        The dataset included patient information such as age, BMI value, smoking status, weekly alcohol consumption, hours of physical activity per day, sleep duration per day, stress level, chronic disease history, and health risk level. I chose the health risk level column as the target variable because the goal was to predict whether a patient was high or low risk and needed early medical intervention.
        I inspected the dataset's structure before preparing it. I identified the number of records, columns, column names, data types, missing values, and duplicate rows. I also examined the number of patients in different age groups and displayed the distribution of health-risk categories. This inspection was important because it provided evidence of the dataset's quality before training the model.
        There are 1,000 records total, which were later split into 600 training records, 200 validation records, and 200 testing records. The training split contained 400 low-risk patient records and 200 high-risk patient records, showing that the target classes were not evenly balanced.
I didn’t find it necessary to remove any of the original fields, so I kept them all. The main cleaning step was handling missing values in the chronic disease column. I replaced missing values with “None” to prevent missing categorical values from interfering with numeric encoding. I also checked for duplicate rows but found none within the dataset.
In addition to using all the original data columns, I created more features to represent relationships that might exist among the patient characteristics. These include:
•	An Age_Group, to categorize patients as young adult, middle-aged, or senior.
•	BMI_Category, to convert BMI values into underweight, normal, overweight, or obese.
•	Is_Active, to show whether patients get at least 2.5 hours of physical activity per week.
•	Age_BMI, shows age multiplied by BMI.
•	Stress_Sleep_Ratio combines stress level and sleep duration.
•	Activity_BMI compares physical activity with BMI.
•	Alcohol_Smoking_Risk combines alcohol consumption and smoking status.
•	High_Risk_Condition identifies patients who have diabetes or heart disease.
•	Lifestyle_Score adds up points for selected lifestyle characteristics, such as adequate sleep, physical activity, lower stress, lower alcohol consumption, acceptable BMI, and not using tobacco.
I created these additional features because a single variable may not fully represent a patient’s overall risk. The lifestyle score summarizes several behavioral factors.
Feature engineering can help a model identify meaningful patterns not immediately represented by the original columns. However, these features still have limitations. Everyone’s body is different. Some people appear to be healthier according to the metrics in the dataset but happen to be at high risk for chronic conditions. Others may have a history of chronic disease, but after learning how to manage their conditions well enough, they are actually low risk for medical intervention. Despite these feature engineering efforts, the model has had difficulty identifying patterns. The relatively low ROC AUC score of 0.51 for the XGBoost cross-validation training indicates that the engineered features did not create a strong separation between the high and not high risk groups.
These processes were appropriate because machine learning models require structured, consistent input data to perform well. This approach reduced the likelihood of errors and improved model reliability. Pandas and Scikit-learn were especially important at this stage because they are commonly used to manage tabular data effectively.
E. Data Analysis Process
1. Methods Used to Analyze the Data
Because the research question requires predicting a category rather than estimating a continuous number, I selected a supervised machine-learning classification model as the analytical approach. Classification is appropriate because the dataset contains patient records with known health-risk labels, allowing the model to learn patterns and predict whether a patient belongs to the high or low risk category. 
I selected XGBoost because the dataset is tabular, with both numeric and categorical information. XGBoost is appropriate because it is commonly used in healthcare analysis. It builds multiple decision trees and combines their results to identify nonlinear relationships.
Originally, the dataset contained multiple health risk categories: low, moderate, and high. I changed the target variable to a binary outcome to focus the analysis on identifying high-risk patients who needed early medical intervention.
2. Advantages and Limitations of the Tools and Techniques
Python and Scikit-learn provide a flexible and popular framework for machine learning projects. XGBoost is a machine learning model that performs well on structured datasets with strong predictive accuracy. Cross-validation helps because it provides a more reliable estimate of model performance than relying on a single train-test split.
One limitation is that these tools require careful preprocessing and tuning. XGBoost is less interpretable than simpler models, which can make it harder to explain specific predictions. In addition, model performance depends heavily on data quality, so features must be engineered carefully to achieve the desired predictive capability.
3. Application of the Analytical Methods
         The following analytical methods were applied to address the research question: “Can patient lifestyle and clinical health data be used to identify those who are high risk and can therefore be targeted for early medical intervention?”
1.	Identify the target variable:
The target variable in the dataset was the “Health Risk Level” column. Originally, there were multiple risk categories, and to focus the analysis on identifying patients who needed medical attention, I converted the target into two risk groups. This supports the research question because the project focuses on identifying high risk patients. It also allowed the use of binary classification metrics such as recall, precision, F1-score, and ROC AUC. However, this approach removed the distinction between low risk and moderate risk patients.
2.	Feature preparation:
I separated the target variable from the input features, and missing values were filled with “None” in the chronic disease column. Removing NULL values allowed the machine learning model to process the column fully.
I then created additional features that represent patient characteristics to prepare for XGBoost modeling. These features were created to give the model additional information about combinations of health and lifestyle factors. However, the selected cutoffs for age, BMI, etc. were chosen at a glance and may not represent the best thresholds for this patient cohort.
3.	Encode remaining variables:
The dataset included other text-based categories such as smoking status, gender, chronic disease history, age group, and BMI category. XGBoost couldn’t use these values, so they were converted into numeric indicators.
4.	Data split
The dataset was split strategically to maintain a similar proportion of high- and low-risk patients in each group. The training set was used to fit the model, while the validation set was used to select the classification threshold. The testing set was not used during training or validation, providing an independent evaluation of the final model.
The split sizes are: 600 training, 200 validation, and 200 testing records. 
5.	Train and tune the XGBoost model:
I implemented RandomizedSearchCV to test different XGBoost settings. The model was trained on 20 possible parameter combinations, and each one was evaluated using five-fold cross-validation. This meant 20 combinations times 5 folds, for a total of 100 model fits. The settings compared included:
•	The number of trees
•	Tree depth
•	Learning rate
•	Feature sampling
•	Row sampling
•	Regularization
•	Class weighting
I found that the best-tested model achieved a ROC AUC score of 0.510, which is just above random guessing. This meant that the model had limited ability to differentiate between high- and low-risk patients.
6.	Evaluate XGBoost performance on the test data:
The model missed most actual high-risk patients when using the standard threshold. After tuning the threshold from 0.50 to 0.20, the model reduced the number of missed high-risk patients from 64 to 14. However, it increased false positives from 20 to 113. This demonstrates the trade-off of identifying more high risk patients. This tuning is justified in healthcare because the consequences of missing a high risk patient are far more negatively impactful than when misclassifying patients who are actually low risk.
7.	Interpret the results against the project objective:
The main objective was to identify high risk patients. This meant recall for high risk patients mattered more than accuracy alone. The standard threshold produced very low recall, correctly identifying only 3 of 67 high-risk patients in the test dataset. Tuning the validation threshold increased observed recall, but it also produced many false positives.
The analytical methods that were used address the research question directly. Supervised classification was appropriate because the project predicted a categorical outcome: whether a patient was classified as high or low risk. XGBoost was selected because it can model complex, non-linear relationships. The validation set and threshold tuning prioritized recall for high risk patients while keeping the test set independent for final evaluation. Five-fold cross-validation assessed whether model performance was reasonably consistent across different subsets of the training data. The final test evaluation showed that lowering the classification threshold increased the number of high risk patients identified, but it also increased false positives. In addition, the ROC AUC results indicated that the model had limited overall ability to distinguish patients who are high risk from low risk using the available features.
F. Results
1. Evaluation of the Output
I evaluated the model output using standard classification metrics. These metrics included accuracy, precision, recall, F1-score, and ROC AUC. Because the project focused on identifying patients classified as high risk, this class was treated as the positive class. The original low and moderate categories were combined into the low risk class.
The following test results were produced by the model at the tuned 0.20 threshold:
Accuracy: 0.365
Precision:0.319
Recall: 0.791
F1-score: 0.455
ROC AUC: 0.456
For comparison, the standard threshold of 0.50 produced:
Accuracy: 0.580
Precision: 0.130
Recall: 0.045
F1-score: 0.067
ROC AUC: 0.456
As mentioned, the tuned threshold substantially improved high-risk recall on the test set. This improvement came with a large increase in false positives. On the other hand, the standard threshold of 0.50 produced nearly 60% accuracy. However, this accuracy came with a recall score of only 0.045, meaning accuracy alone would overstate the model’s usefulness.
The ROC AUC score is independent of the set threshold, so it remains at 0.456 in both sets of results. The identical ROC AUC shows that changing the threshold shifted the operating point but did not improve the model's underlying ability to distinguish between the two classification groups.
Although the tuned threshold produced a higher observed recall score, statistical significance was not tested. Therefore, the analysis cannot establish that the tuned threshold is statistically superior to the standard threshold. It only shows that the tuned threshold better matched the project’s objective of identifying high risk patients in this particular dataset.
Threshold selection heavily influenced the model’s performance. The standard threshold missed most high-risk patients, while the tuned threshold identified nearly 80% of them but misclassified 113 low-risk patients. This means the tuned model should not be used as an autonomous decision-support tool in a clinical setting. The weak ROC AUC, wide uncertainty due to a limited sample size, and a large number of false positives reduce confidence in the model’s effectiveness.
Because additional external validation was not performed, the results cannot establish that the model would perform similarly on a larger or different healthcare dataset. Any claims about clinical usefulness cannot currently be made.
2. Practical Significance of the Solution
While the model captures nearly 80% of high risk patients, the results show severe practical limitations. The solution is not suitable as an autonomous decision support tool. In an active clinical environment, flagging more than half of incoming patients for high-risk treatment would overwhelm medical staff. However, the solution can serve as a conditional pre-triage filter. It could flag high risk patients for standard preventive actions like questionnaires, call-backs, or diet and exercise recommendations.
         Before practical deployment in a live setting, the dataset should include additional features like blood pressure readings, HbA1c panels, etc., to hopefully improve class separation and reduce the false-positive rate.
3. Overall Success and Effectiveness of the Project
         The completed project successfully implemented an end-to-end analytics workflow. The process included importing a healthcare dataset, inspecting and preparing the data, creating different datasets, training an XGBoost classification model, and evaluating the results using multiple performance measures.
         The main objective was to identify high risk patients. Because misclassifying a high-risk patient is more costly than misclassifying a low-risk patient, recall was treated as a very important measure of effectiveness.
         The project effectively demonstrated a complete machine-learning workflow, but the model was only partially effective for healthcare risk prediction. The model could serve as a preliminary screening aid, with flagged patients completing an additional questionnaire to explore their health further. The model should not be used as an autonomous decision-support tool, as its predictions are not yet reliable enough.
G. Key Takeaways
1. Conclusions from the analysis
         This project investigates whether clinical and lifestyle data can reliably identify patients at high health risk. The XGBoost classification analysis showed that lifestyle and clinical data alone provided limited predictive capability for health risk classification.
         Class separation was weak at the base model threshold. This resulted in clinically unacceptable scores for accuracy, recall, precision, F1-score, and ROC AUC.
There were limitations in the analysis. For example, the dataset used many self-reported lifestyle behaviors such as hours of physical activity, alcohol consumption, sleep duration, and stress levels. The model may have performed better if it had included critical physiological biomarkers known to be tied to chronic diseases as features, such as resting heart rate, family medical history, and blood pressure readings.
The analysis was also restricted to 1,000 total observations in the dataset, which was partitioned into 3 sets. A small number of prediction changes would substantially affect the reported recall, precision, and F1-score. It wouldn’t be rational to assume that the measured performance in this project would remain the same on a larger patient population.
2. Tools used for storytelling
The tools used in this project support effective storytelling by turning model results into easy-to-understand visuals. Python visualization libraries like Matplotlib and Seaborn create clear, professional charts that highlight model performance. The confusion matrix shows how the model classifies patients, while the ROC curve shows how well it distinguishes between the two risk categories. These visuals make the results more accessible to technical and non-technical audiences and support the view that healthcare professionals treat patients as precious “Lives,” not just numbers.
3. Recommended Courses of Action
My first recommendation is to use the model as a preliminary screening tool to identify patients who might benefit from early medical intervention. This would help healthcare providers focus resources on higher-risk individuals.
Different lifestyle and health data patterns emerge as we explore different parts of the world. Different cultures and living conditions mean we cannot train models like this to be all-encompassing. If we want to implement predictive models in healthcare clinics, I recommend using training data from the same clinic that will implement the technology.
Additionally, I recommend improving the model in future work by testing additional datasets, refining features, and comparing more available machine learning algorithms. This will further improve predictive capabilities and increase provider confidence in using the model as a decision support tool.


















Sources
Abburi, A. (n.d.). Healthcare Real-Time Risk Dataset. Kaggle. https://www.kaggle.com/datasets/alekhyaabburi/healthcare

