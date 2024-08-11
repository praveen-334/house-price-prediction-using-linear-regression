
# House Price Prediction using Linear Regression

This project uses a linear regression model to predict house prices based on various features such as number of bedrooms, bathrooms, square footage, etc. The model is trained on the house price data provided in the house-price-data.csv file.


## Dependencies


Python 3.x

NumPy

Pandas

Scikit-learn


# Data preprocessing

### 1 . Load the CSV file:

```python
  data = pd.read_csv('house-price-data.csv')

```

### 2 . Handle missing values (if any) and perform feature engineering as needed.


### 3 . Split the data into training and test sets:
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

```
# Model Training
### 1 . Create and train the linear regression model:
```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)


```
### 2 . Evaluate the model's performance on the test set:

```python
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse:.2f}')
print(f'R-squared: {r2:.2f}')

```

# Future Improvements

Explore more advanced regression techniques, such as polynomial regression or regularized models, to potentially improve the model's performance.
Conduct feature selection to identify the most important predictors of house prices.
Gather additional data sources to enrich the feature set and capture more aspects that influence house prices.

https://github.com/praveen-334/house-price-prediction-using-linear-regression/pull/1#issue-2430583429


