"""
--------PREDICTION OF CAR MILEAGE---------

"""

import pandas
from sklearn.linear_model import LinearRegression
import seaborn as sns

filename='auto-mpg.csv'
names=['mpg','cyl','disp','hp','wt','acc','my','ori','name']
dataframe=pandas.read_csv(filename,names=names)
dataframe.dropna(inplace=True)
array=dataframe.values

X=array[:,[1,2,3,4,5,6,7]]
Y=array[:,0] # first row shows the miles per gallon (mpg)
model=LinearRegression()
results=model.fit(X,Y)

print("the coefficients are")
print(results.coef_)
print()
print("the intercept is")
print(results.intercept_)    
print()
print("score is")
print(round(results.score(X,Y),5))
print("%.3f%%" % (results.score(X,Y)*100.0))
print()
ax=sns.barplot(x="cyl",y="mpg",hue='ori',data=dataframe);
ax.set(xlabel='number of cylinders',ylabel='mpg') #barplot

predicted=model.predict(X)

y_predicted=0
for i in range(392): #Calculation for root mean square
    y_predicted+=(Y[i]-predicted[i])**2
    predicted[i]=Y[i]-predicted[i]
y_predicted=y_predicted/392

s=pandas.DataFrame(predicted,columns=['Error for each rows'])

print()
print("--description of the errors in predicted value of each row---")
print()
print(s.describe())
print( "RMS VALUE:%.3f" % y_predicted**(0.5)) # ** means power
print()
dataframe.plot(kind='box' , subplots=True,layout=(2,6)) #boxplot

from pandas.tools.plotting import scatter_matrix
scatter_matrix(dataframe)
