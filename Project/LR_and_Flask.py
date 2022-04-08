
import pandas as pd

df = pd.read_csv("Ex__Data/ex_data.csv").set_index("date")
df.corr()[['temp']].sort_values('temp')

predictors = ['temp_1', 'temp_2', 'temp_3',
              'temp_min_1', 'temp_min_2', 'temp_min_3',
              'feels_like_1', 'feels_like_2', 'feels_like_3',
              'humidity_1', 'humidity_2', 'humidity_3',
              'pressure_1', 'pressure_2', 'pressure_3',
              'temp_max_1', 'temp_max_2', 'temp_max_3']
df2 = df[['temp'] + predictors]

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# manually set the parameters of the figure to and appropriate size
plt.rcParams['figure.figsize'] = [16, 22]

# call subplots specifying the grid structure we desire and that
# the y axes should be shared
fig, axes = plt.subplots(nrows=6, ncols=3, sharey=True)

# Since it would be nice to loop through the features in to build this plot
# let us rearrange our data into a 2D array of 6 rows and 3 columns
arr = np.array(predictors).reshape(6, 3)

# use enumerate to loop over the arr 2D array of rows and columns
# and create scatter plots of each meantempm vs each feature
for row, col_arr in enumerate(arr):
    for col, feature in enumerate(col_arr):
        axes[row, col].scatter(df2[feature], df2['temp'])
        if col == 0:
            axes[row, col].set(xlabel=feature, ylabel='temp')
        else:
            axes[row, col].set(xlabel=feature)
#plt.show()
# import the relevant module
import statsmodels.api as sm

# separate our my predictor variables (X) from my outcome variable y
X = df2[predictors]
y = df2['temp']

# Add a constant to the predictor variable set to represent the Bo intercept
X = sm.add_constant(X)
X.iloc[:5, :5]
# (1) select a significance value
alpha = 0.05

# (2) Fit the model
model = sm.OLS(y, X).fit()

# (3) evaluate the coefficients' p-values
print(model.summary())
X = X[['const', 'temp_min_1', 'temp_min_2', 'temp_min_3', 'feels_like_1', 'feels_like_2', 'feels_like_3', 'temp_max_1']]
model = sm.OLS(y, X).fit()
print(model.summary())
from sklearn.model_selection import train_test_split

# first remove the const column because unlike statsmodels, SciKit-Learn will add that in for us
X = X.drop('const', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)
from sklearn.linear_model import LinearRegression

# instantiate the regressor class
regressor = LinearRegression()

# fit the build the model by fitting the regressor to the training data
regressor.fit(X, y)

# make a prediction set using the test set
prediction = regressor.predict(X_test-273)
print(prediction)
y_pred = regressor.predict(X_test)
df = pd.DataFrame({'Actual': y_test-273.15, 'Predicted': y_pred-273.15})
print(df)
# Evaluate the prediction accuracy of the model
from sklearn.metrics import mean_absolute_error, median_absolute_error
prediction = regressor.predict(X_test-273)
print("The Explained Variance: %.2f" % regressor.score(X_test, y_test))
print("The Mean Absolute Error: %.2f Kelvin" % mean_absolute_error(y_test, prediction))
print("The Median Absolute Error: %.2f Kelvin" % median_absolute_error(y_test, prediction))


days = ['11th April', '12th April', '13th April', '14th April', '15th April', '16th April', '17th April', '18th April', '19th April', '20th April', '21th April', '22th April', '23th April', '24th April', '25th April', '26th April', '27th April', '28th April', '29th April', '30th April', '1th May', '2th May', '3th May', '4th May', '5th May', '6th May', '7th May', '8th May', '9th May', '10th May']
predic = prediction
predic_dic = {days[i]: predic[i] for i in range(0, 30)}
print(predic_dic)


from flask import Flask, redirect, url_for,render_template,request


app = Flask(__name__)
# flask constructor, takes the name of the current module as argument


@app.route('/')
def first():
    return render_template("first.html")


@app.route("/second")
def second():
    return render_template("second.html", temp=predic_dic, a=regressor.score(X_test, y_test), b=mean_absolute_error(y_test, prediction), c=median_absolute_error(y_test, prediction))



    app.run(debug=True)



