# Regression
## Problems
Real Estate Price Prediction
Student Mark Prediction

## Notes
Check data quality. data.isnull().sum()
Select Features
    Histogram - Check the distribution of one property only.
    Scatter Plot - Check one feature at a time v/s output
    correlation_matrix = data.corr() + heatmap.
        Value near to 1 or -1 are more related to output.

Now 2 DF X and y
train_test_split()

Choose model > fit() // train > predict > check with actual value through scatter graph.

Calculate MAE and R²
the lowest MAE and the highest R², making it the best-performing model among those evaluated.
the highest MAE and the lowest R², indicating it may be overfitting to the training data and performing poorly on the test data



# Classification
## Problems
Add Click Through Rate Prediction
Language Detection - We generate the vectors for each sentence.
    cv = CountVectorizer()
    X = cv.fit_transform(x)


## Notes
Convert integer into category value 0-No and 1-Yes.

Box chart for each feature with categories. So, we can have a range of feature's value for user go for these categories.

Now 2 DF X and y
train_test_split()

Choose model > fit() // train > predict > check with actual value through scatter graph.

from sklearn.metrics import accuracy_score
print(accuracy_score(ytest,y_pred))

Multinomial Naïve Bayes algorithm to train the language detection model as this algorithm always performs very well on the problems based on multiclass classification.





# Clustering Analysis

# Deep Learning

## Notes
Read Inputs.
Train and Test Data

Train = Train + Validate

Decide Input (Size) Layer > Hidden Layer > Output Layer
    I/P and O/P count, activation function

Train - Loss function; Optizier Option, Metrics.

It predict probability of all category. Choose maximum one
# Recommendation System

# Time Series

# End to End Project
## Chatbot
Define Intents
Create training data
Train the chatbot
Build the chatbot
Test the chatbot
Deploy the chatbot