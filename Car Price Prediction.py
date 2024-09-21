
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import skew
#%matplotlib inline

 
import matplotlib.pyplot as plt
plt.style.use("ggplot")
plt.rcParams['figure.figsize']=(12,8)

 
data=pd.read_csv("Cardheko.csv")

 
data.head()

 
data.info()

 
#PLOTTING INDEPENDANT AND DEPENDANT VARIABLE
sns.pairplot(data,x_vars=['year','kmdriven','fuel','sellertype','transmission','owner'],y_vars='sellingprice',height=3,aspect=0.7)

 
#This shows that the year and the Kilometer driven affects the price.

 
object_columns = data.select_dtypes(include=['object']).columns.tolist()

 
object_columns

 
data.info()

 
from sklearn.preprocessing import LabelEncoder

for column in object_columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])

 
sns.heatmap(data.corr(),annot=True)

 
dropdata=data.drop(['name','fuel','sellertype','transmission'],axis='columns')
dropdata

 
#linear Regression
from sklearn.linear_model import LinearRegression
x= data[['year','kmdriven','sellingprice']]
y=data.owner
model=LinearRegression()
model.fit(x,y)

print("intercept=",model.intercept_)
print("coefficient=",model.coef_)
list(zip(['year','kmdriven','owner'],model.coef_))

 
pred=model.predict([[2015,85422,350000]])
print(pred)


