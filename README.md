# Linear_Regression
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

For the last line I made a small change to the df to plot it.

LN[5]
Manuel one hot encoder 
