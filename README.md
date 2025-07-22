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
So in the first version that I built this one is one x variable only

Slope function:
For this function we use this formula 
∑ - sum of 
^2 - squared 
m = (n∑xy - ∑x∑y) / (n∑x^2 - (∑x)^2)

Working as 
