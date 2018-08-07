# Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV

# Loading the Data Set
df = pd.read_csv('ml_house_data_set.csv')

# Remove the Fields from the Data Set that we don't want in our Model
del df['house_number']
del df['unit_number']
del df['street_name']
del df['zip_code']

# Replace the Categorical Data with One-Hot Encoded Data
features_df = pd.get_dummies(df, columns=['garage_type', 'city'])

# Remove the Sale Price from the Feature Data
del features_df['sale_price']

# Let's view our Data Set
features_df.head()

# Create the "X" and "y" Arrays
X = features_df.as_matrix()
y = df['sale_price'].as_matrix()

# Split the data set in a training set (70%) and a test set (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Fit regression model
model = ensemble.GradientBoostingRegressor(
    n_estimators=1000,
    learning_rate=0.1,
    max_depth=6,
    min_samples_leaf=9,
    max_features=0.1,
    loss='huber',
    random_state=0
)
model.fit(X_train, y_train)

# Save the trained model to a file so we can use it in other programs
joblib.dump(model, 'trained_house_classifier_model.pkl')

# Find the error rate on the training set
mse = mean_absolute_error(y_train, model.predict(X_train))
print("Training Set Mean Absolute Error: %.4f" % mse)

# Find the error rate on the test set
mse = mean_absolute_error(y_test, model.predict(X_test))
print("Test Set Mean Absolute Error: %.4f" % mse)


# For realistic work delete/replace the above Code
# From line 31 until line 55
# With the code below
# But with the code below you will have to wait
# For A LONG TIME to get Results

"""

# Split the Data Set: Training Set = 70%, Test Set = 30%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Fit Regression Model
model = ensemble.GradientBoostingRegressor()

# Parameters we want to try
param_grid = {
  'n_estimators': [500, 1000, 3000],
  'learning_rate': [0.1, 0.02, 0.02, 0.01],
  'max_depth': [4, 6],
  'min_samples_leaf': [3, 5, 9, 17],
  'max_features': [1.0, 0.3, 0.1],
  'loss': ['ls', 'lad', 'huber']
}

# Define the grid search we want to run. Run it with four cpus in parallel.
gs_cv = GridSearchCV(model, param_grid, n_jobs=4)

# Run the grid search - on only the training data!
gs_cv.fit(X_train, y_train)

# Print the parameters that gave us the best result!
print(gs_cv.best_params_)

# Save the trained model to a file so we can use it in other programs
joblib.dump(gs_cv, 'trained_house_classifier_model.pkl')

# Find the Error Rate on the Training Set
mse = mean_absolute_error(y_train, gs_cv.predict(X_train))
print('Training Set Mean Absolute Error: %.4f' % mse)

# Find the Error on the Test Set
mse = mean_absolute_error(y_test, gs_cv.predict(X_test))
print('Test Set Mean Absolute Error: %.4f' % mse)

"""


