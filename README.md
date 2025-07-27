# Linear_Regression from scratch 
Built Linear Regression using the math behind it and compared it to SKLearns version 

Data set used: https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression

ln[1] 
In this cell I import important third-party libraries that will be used all over this project.

Matplotlib.pylot(plt) will be used to visualize and plot data/results

Numpy(np) used for most of the math processes in the MLR as it is faster than python(due to it using C and Fortran)

Pandas(pd) is used for analyizing, cleaning, and exploring data. Able to convert files into sortable readable data to use for training. 

Pandas.api.types I used this for my manuel one hot encoder. 

ln[2]
X,y very basic data set I created in order to test my simple linear regression model. 

Now for the actual linear regression this is the one that is simple. I built it by just watching a yt video expaining the math for it. 

In my initialization I set all the variables that will be used globally in the class. 

Explanation for all var: 
self.x / y = x / y just the x and y values being passed through so the model knows them. 

self.n the length of the array X ( could change with y since they're both going to be equal) 

self.sum_x all of the x values in the x array added up with each other so in this example it would be 1+2+3+4+5+6 = 21 
self.sum_y same concept as sum_x 
self.sum_xy each x and y value multiplied together so 1 * 6 etc then added together 1*6 + 2*9 + 3*12 etc 
self.sum_x2 all the x values to the power of two added together so 1^2 + 2^2 + 3^2 etc 

Simple linear regression explanation: 
It will make the classic equation - Y = b + mx
Y - dependent
x - independent
b - intercept
m - coefficient 

Slope function:
For this function we use this formula 
∑ - sum of 
^2 - squared 
m = (n∑xy - ∑x∑y) / (n∑x^2 - (∑x)^2)

Working as its essentially another version of y/x 

Y-intercept function:
b = ∑y - m*∑x / n

LN[3]
This cell we test the model with the data given to see how accuracte the model is to it. 
Initializing it with the data then coming up with the slope and y-int with the functions we made. 

y_pred we iterate through every x value multiplying it by the slope and then adding the y-intercept to make the famous equation y = mx + b 

Then I plot it to see what the results were(IDK why it doesn't show up here but feel free to try it out for yourself I promise it works) Using matplotlib to represent my predictions on a graph. Scatter to represent real data then predicted line to represent the equation I derived from the data itself. 

LN[4]
data = pd.read_csv("student_scores.zip")
I used this to make the zip into a readable dataset to train my model on.  

Next is a custom built standardization function based on:
Equation: z = (x - μ) / σ
σ (sigma)- is used to represent the stanard deviation of the population
μ (mu) - is used to represent the mean of the population
x - individual value 
To do this I iterated through each item in each column of the df calculating the std and means of the column before hand. 

For the last line I removed the non-numerical value of the df in order plot/visualize the data.

LN[5]
In this cell I build a manual one hot encoder, which makes non-numerical data into 0/1 to represent the values. An example would be like the Extracurricular Activites where instead of being a yes or no we use a 1 for yes and 0 for no. 

Explanation of the code: 
As said earlier I imported is_numeric_type for one reason and it was this to check if the columns values were a number or not. So as we iterate through the df's columns the encoder checks for val is number. If it is the function returns the column, however if it isn't there's a whole new process. In the function I set a unique_vals to all the unique values of the non-numeric column. So this one will only have 2 values "yes" and "no". And if that is the case with only 2 values I go to the if function and then assign it only 0/1 iterating through each value of the column and returning it. But the amount of values is greater than 2 we return a long number of 0s and 1s representing the non-numerical value passed through the function. 

LN[5]
When running the programm initially and making this project a big flaw I had was if the values in the data set weren't a float the linear regression would be not even close to the actual values. So I made a simple function that makes all the values of the df a float. 

LN[6]
Now we get into data visualization with box plots! Iterating through each value to get a sense of data's values and to see if there's any outliers. 

Box Plot: 

A box plot splits the data into four equal parts (quartiles). The box shows the middle 50% of values, from the first quartile (Q1) to the third quartile (Q3), with a line in the center marking the median. The “whiskers” extend to the lowest and highest values within 1.5×IQR of the quartiles. Any extreme values beyond this range are plotted as outliers to avoid skewing the visualization.

<img width="326" height="155" alt="image" src="https://github.com/user-attachments/assets/5d49c26d-0ea1-4443-9e8a-8f1c7694ee40" />

LN[7]

Here I plot all the values in comparision to each other in a scatter plot trying to find the positive and negative correlations. And in this dataset we see only positive correlations. 

<img width="326" height="326" alt="image" src="https://github.com/user-attachments/assets/e8a23d20-9258-445e-9c51-9c92f1664c53" />

LN[8]

In cell 8 I graph a heat map which shows how much each variables correlate with each other. Before the calculations are made I standardize the data to ensure all features are on the same scale. Ensuring that features with large numerical values don't overpower the calculations. 

<img width="326" height="326" alt="image" src="https://github.com/user-attachments/assets/8bf9dff5-131b-4ae2-8b0d-eee98c886218" />

LN[9]

This cell was made to ensure that the things I had built were functioning. 

LN[10]

Manual Multi-Linear regression:
For this we have multiple independent varibles go through and predict one dependent variable.

__init__(self,df):
I make the class and initialize key variables with a dataframe being passed in. 
self.x/self.y seperate all independent variables from dependent. Then I split data into train(x/y) and test(x/y) 80:20 split. Then I added straight 1s for the first column of the X train/test in order to represent an intercept 

Beta(self):
 ˆβ = (XᵀX)⁻¹XᵀY
 
Xᵀ - inverse the Matrix so ex:

[[2,2,3], -> tranpose -> [[3,2,2],
[2,6,7]]                [7,6,2]]

()⁻¹ - inverse 

[[4, 7], -> [[ 0.6 -0.7],
[2, 6]]     [-0.2  0.4]]

This is the training part of the model in order to get the proper coeffients for testing.

predict(self): 

Multiply the independent values by coefficients to get a score prediction

se_beta(self,prediction): 

Finding the standard Error of the beta coefficient: 

D - diag of inverse of self.xᵀ * self.x

How does diagonal work? 

[[2,3,4],                     
 [3,4,2], --> diag function --> [2,4,4] 
 [6,7,4]]

 y residual - ŷ - subtracting the predicted y values from the actual ones calculated 

 resSum - sum of all the ŷ ^ 2
Ex: [2,3,4,5] -> resSum -> 54

Se(standard error) - ( ŷ / (number of rows - number of columns))

Then to get the standard error of the beta we multiply the standard error by the cdiag that we had calculated before hand. 

This is our first evaluation method. There's two more that I used R^2 value and MSE.

evaluate(self,predictions):

r2_score - the closer to 1(neg or pos) means the model is more accurate. R² quantifies how much of the variability in the outcome (dependent) variable is explained by the model's predictions. So if you have a score of .7 that means that 70% of the variation is accounted for by the model by 30% is unexplained. 

Mean Standard Error - the precision of the model the smaller the number the better it is. 

My model scored
R^2: .9890
MSE: 4.0826

sklearn scored: 
R^2: .9889
MSE: 4.08262

Meaning that my model is now industry standard! 
