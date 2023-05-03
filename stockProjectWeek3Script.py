# Before you continue, make sure to understand what a "function" is: https://www.youtube.com/watch?v=asvJ2ly7efM
# Make sure you understand what a linear equation (a type of function) is:
# https://www.khanacademy.org/math/algebra/x2f8bb11595b61c86:forms-of-linear-equations/x2f8bb11595b61c86:intro-to-slope-intercept-form/v/slope-intercept-form
# Make sure you understand what a logarithm is (it is a function: https://simple.wikipedia.org/wiki/Logarithm)
# Also make sure you understand what is GDP per capita (the amount of things are produced per person in a given place:
# https://simple.wikipedia.org/wiki/Gross_domestic_product)

# This here is a command that give us commands allowing us to create representations of lists of numbers (arrays) like
# 1, 2, 3 4 or 5, 6, 7, 8.
# numpy is a "library" of commands we can get access to, and here we give it the nickname "np".
import numpy as np
# This here is library of commands allowing us read data from files.
import pandas as pd
# This here is a library of commands allowing us to draw graphs and put points on them given x-coordinates and
# y-coordinates.
import matplotlib.pyplot as plt
# This here is a library of commands (LinearRegression) coming from another library (linear_model, in sk_learn).
# This one allows us to draw lines of best fit (represented by linear equations) through points on graphs.
# This library allows us to create "models" to make predictions (this will be explained later), specifically the
# "LinearRegression" model.
from sklearn.linear_model import LinearRegression

# Today we will be looking at a country called Brazil (https://simple.wikipedia.org/wiki/Brazil).
# We will be looking at
# * the logarithim (log) of GDP per capita in Brazil
# * on what scale from 0 to 10 do Brazilians rate their quality of life (Life Ladder values) and
# * how these looked like in Brazil from the year 2005 to the year 2022.

# We might wonder: Is there a linear equation that could more or less pass through the points on a graph made from:
# * putting the log of GDP per capita of Brazil on the x-axis (our independent variable) for each given year and
# * for the same year putting the Life Ladder values on the y-axis (our dependent variable)?

# To see, we need to make a graph.

# This command finds and gives a name for all our data from 2005 to 2022 about Brazil's per capita GDP and its Life
# Ladder values.
# It is a function named "read_csv" from the pandas library reading some CSV file's location as an input.
# We put its content into a variable we call "df".
# It is a big collection of data with some things we are not interested in, but we will get what we want.
# The file location is a sequence of characters, also called a "string".
df = pd.read_csv('stockProjectWeek3Data.csv')

# This command puts our x-coordinates in an array (from the NumPy library) named "x".
# An array is collection of objects (ex: here we have numbers, those being our x-coordinates) in a certain order.
# We get this part of the data from our bigger collection of data using the names of the things we want in square
# brackets.
x = np.array(df['Brazil Log GDP per Capita'])
# We modify x by only keeping the first 17 numbers contained in x.
# We do this to get rid of empty (invalid) parts of our supplied data.
# More technically, we start at 0 and count up to 1, 2, 3, ..., 16, which is right before we reach 17.
# Here, 0 represents the 1st contained number within the original array x, 1 represents the 2nd contained number and 2
# represents the 3rd contained number, etc.
# We turn x into an array with those contained numbers, in an order corresponding to 0, 1, 2, ..., 16
x = x[:17]

# This command puts our y-coordinates in an array, named "y"
# The location of each y-coordinate value within this array matches the position of the corresponding coordinate in the
# x-coordinate array.
y = np.array(df['Brazil Life Ladder (meaure of quality of life on scale from 0-10)'])
# Again, only the first 17 numbers are valid to match the number of x-coordinates, so we only keep those.
y = y[:17]

# This command from the Matplot library puts our years as dots on a graph, with:
# * the dot's position along the x-axis coming from the x-coordinate array and
# * the dot's position along the y-axis being that from the corresponding place in the y-coordinate array
# It is a function named "scatter" from the plt library that takes in two "arguments" (input variables), those being our
# arrays named "x" and "y".
# We match the x- and y-coordinates together to determine the location of our dots.
# We will be adding and changing things in this graph as we go.
plt.scatter(x, y)

# This command will determine what our x-coordinates represent on the drawn graph, that being the logarithm of GDP per
# capita of Brazil for a given year.
# It is a function called "xlabel" that takes in a string as input and as output it will put the string on the right
# place on the graph's x-axis when it is drawn.
plt.xlabel('Log GDP per capita in Brazil')

# This command will determine what our y-coordinates represent on the drawn graph, that being the rated quality of life
# according to the Life Ladder system in the same year.
# It is similar to the one above, except it will affect the y-axis instead.
plt.ylabel('Quality of life rating in Brazil (Life Ladder value)')

# We can see our work-in-progress graph with this command.
plt.show()

# Here we specify a type of "model" from the sk_learn library for our dots, which is basically a set of information
# which will include:
# * a straight line (the "Linear" in "LinearRegression") of best fit created for our graphed dots
# * and some extra information we will look at later (a lot of things related to "Regression")
model = LinearRegression()

# This command actually creates a "model" using our dots (from their coordinates in the arrays).
# Note: When we include our array "x", we invoke a method (similar to a function) called "reshape" to make sure "fit"
# (also a method) likes its format
model.fit(x.reshape(-1, 1), y)

# Here we prepare to make a graph again with our dots.
plt.scatter(x, y)

# This command will ensure that the graph includes our line of best fit, with a red coloration.
# The line of best fit is a function represented by a method here which we call "predict":
# * As input it takes in x-coordinates (logarithm of Brazil's GDP per capita for a year)
# Note "reshape" is used here again for our array x.
# * As output it gives back y-coordinates (Life Ladder value in the same year) which would be exactly on the line.
# The y-coordinates of the dots can be called the ACTUAL Life Ladder values for the corresponding logarithm of
# Brazil's GDP per capita, while the y-coordinates on the line of best fit can be called the PREDICTED Life Ladder
# values for the corresponding logarithm of Brazil's GDP per capita.
plt.plot(x, model.predict(x.reshape(-1, 1)), color='red')

# Here we add axis labels again.
plt.xlabel('Log GDP per capita in Brazil')
plt.ylabel('Quality of life rating in Brazil (Life Ladder value)')

# This command displays the final resulting graph:
plt.show()

# Our "model" can also let us know how well it thinks the line of best fit is able to fit the dots.
# One way it can let us know is through a value called R^2 (R squared), also known as the coefficient of determination.
# We get it through the "score" method and assign it to the variable r_squared. Once again, we use "reshape".
r_squared = model.score(x.reshape(-1, 1), y)
# For our purposes it can be anywhere from 0 to 1:
# * Values closer to 1 indicate better fits (exactly 1 is an exactly perfect fit).
# * Values closer to 0 indicate worse fits.
# The worse the fit, the more the line of best fit (a function) is likely not a good choice if we want to make
# predictions about the value of a y-coordinate given a certain value for the x-coordinate of the corresponding dot.
# (Another way to define R^2: it is the proportion of variance or variability in the dependent variable that is
# explained by, or appears to have a correspondence to different values of, the independent variable)
# If it seems that you cannot make any straight lines or curves that fit the dots or otherwise predict their approximate
# positions in a logical shape or pattern, it is likely that there is no relationship between the independent and
# dependent variables.
# Let's see what the R^2 value is for our line of best fit using the print function
# (the string 'R-squared: ' is separate piece of text that will appear by our number, stored in r_squared):
print('R-squared: ', r_squared)

# Looks like our result for R^2 is around 0.10. In practice, we would want it to be 0.7 or larger if we wanted to
# consider it a good fit, and in our case our R^2 indicates that our straight line is probably not a good fit.
# Recalling our dependent and independent variables, this means we lack convincing evidence for there being a
# relationship between Brazilians' quality of life in terms of Life Ladder ratings ad the log of Brazil's GDP per
# capita from the years 2005 to 2022.

# Here we don't have anything else to do. Bye bye!
