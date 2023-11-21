import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load data
cardata = pd.read_csv('cars.csv')

# Data cleaning and handling outliers
data = cardata.drop(['Model'], axis=1)
data_no_rv = data.dropna(axis=0)

q = data_no_rv['Price'].quantile(0.99)
data_price_in = data_no_rv[data_no_rv['Price'] < q]

q = data_no_rv['Mileage'].quantile(0.99)
data_mileage_in = data_no_rv[data_no_rv['Mileage'] < q]

# Checking linear regression assumptions and transforming the target variable
data_cleaned = data_mileage_in.reset_index(drop=True)
log_price = np.log(data_cleaned['Price'])
data_cleaned['log_price'] = log_price

# Training the model
targets = data_cleaned['log_price']
inputs = data_cleaned.drop(['log_price'], axis=1)

# Scaling the data and splitting into train/test sets
scaler = StandardScaler()
scaler.fit(inputs)
inputs_scaled = scaler.transform(inputs)
x_train, x_test, y_train, y_test = train_test_split(inputs_scaled, targets, test_size=0.2, random_state=365)

# Linear regression model training
reg = LinearRegression()
reg.fit(x_train, y_train)
y_hat = reg.predict(x_train)
y_hat_test = reg.predict(x_test)

# Model evaluation
print('Training set metrics:')
print('R-squared:', r2_score(y_train, y_hat))
print('Mean Absolute Error:', mean_absolute_error(y_train, y_hat))
print('Mean Squared Error:', mean_squared_error(y_train, y_hat))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_train, y_hat)))

print('\nTest set metrics:')
print('R-squared:', r2_score(y_test, y_hat_test))
print('Mean Absolute Error:', mean_absolute_error(y_test, y_hat_test))
print('Mean Squared Error:', mean_squared_error(y_test, y_hat_test))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_hat_test)))

# Visualization
f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(15, 3))
ax1.scatter(data_cleaned['Year'], data_cleaned['Price'])
ax1.set_title('Price and Year')
ax2.scatter(data_cleaned['EngineV'], data_cleaned['Price'])
ax2.set_title('Price and EngineV')
ax3.scatter(data_cleaned['Mileage'], data_cleaned['Price'])
ax3.set_title('Price and Mileage')
plt.show()

plt.scatter(y_train, y_hat)
plt.xlabel('Targets (y_train)', size=18)
plt.ylabel('Predictions (y_hat)', size=18)
plt.xlim(6, 13)
plt.ylim(6, 13)
plt.show()

sns.displot(y_train - y_hat)
plt.title("Residuals PDF", size=18)
plt.show()

plt.scatter(y_test, y_hat_test)
plt.xlabel('Targets (y_test)', size=18)
plt.ylabel('Predictions (y_hat_test)', size=18)
plt.xlim(6, 13)
plt.ylim(6, 13)
plt.show()

sns.displot(y_test - y_hat_test)
plt.title("Test Set Residuals PDF", size=18)
plt.show()
