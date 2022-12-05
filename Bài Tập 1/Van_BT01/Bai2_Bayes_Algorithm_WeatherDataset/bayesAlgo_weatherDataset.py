#import all the packages
import cols as cols
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
import sklearn.metrics
import statsmodels.api as sm
import plotly.express as px #for plotting the scatter plot
import seaborn as sns #For plotting the dataset in seaborn
sns.set(style='whitegrid')
import warnings
warnings.filterwarnings('ignore')

#Read the excel file
data=pd.read_csv('weather.csv')
print(data.head(5))
print(data.describe())

# checking if data imputation is required:
print(data.columns)
data.isnull().any()
# No null value is present in the dataset but if there was one we could've used:
# data = data.dropna(axis = 0, how ='any')
data=data.drop(['day'],axis=1)

le = LabelEncoder()
data['outlook_encoded']= le.fit_transform(data['outlook'])
data['temp_encoded']= le.fit_transform(data['temp'])
data['humidity_encoded']= le.fit_transform(data['humidity'])
data['wind_encoded']= le.fit_transform(data['wind'])
data['play_encoded']= le.fit_transform(data['play'])
print(data.head(5))

# heatmap of correlation matrix with annotations in 2 different shades
cols=['outlook_encoded','temp_encoded','humidity_encoded','wind_encoded','play_encoded']
cor=data[cols].corr()
hm1 = sns.heatmap(cor, annot = True,cmap='YlGnBu')
hm1.set(xlabel='\nFeatures', ylabel='Features\t', title = "Correlation matrix of data\n")
plt.show()

features=['outlook_encoded', 'temp_encoded', 'humidity_encoded','wind_encoded']
x=data[features]# since these are the features we take them as x
y=data['play_encoded']# since play is the output or label we'll take it as y
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2,random_state=1)
print("\nShape of x_train:\n{}".format(x_train.shape))
print("\nShape of x_test:\n{}".format(x_test.shape))
print("\nShape of y_train:\n{}".format(y_train.shape))
print("\nShape of y_test:\n{}".format(y_test.shape))

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
scaled_x_train = sc.fit_transform(x_train)
scaled_x_test = sc.transform(x_test)
print(x_train)
print("____________________________________________________________________________")
print("",scaled_x_train)
## Before
# View the relationships between variables; color code by play
before= sns.pairplot(data[cols], hue= 'play_encoded')
before.fig.suptitle('Pair Plot of the dataset Before scaling', y=1.08)
## After:
data2= pd.DataFrame(data= np.c_[scaled_x_train, y_train],
columns= features + ['play_encoded'])
after= sns.pairplot(data2, hue= 'play_encoded')
after.fig.suptitle('Pair Plot of the dataset After scaling', y=1.08)

#--------------------------training the model----------------------------------
x_train=scaled_x_train
x_test=scaled_x_test
model = GaussianNB()
model.fit(x_train, y_train)
y_prediction= model.predict(x_test)
report=pd.DataFrame()
report['Actual values']=y_test
report['Predicted values']= y_prediction
print(report)

#--------------------------model evaluation----------------------------------
ConfusionMatrix=confusion_matrix(y_test,y_prediction)
print(ConfusionMatrix)
ax=sns.heatmap(ConfusionMatrix,annot=True,cmap="YlGnBu")
ax.set_title('Confusion matrix')
ax.set_xlabel('Predicted values')
ax.set_ylabel('Actual values')
#Ticket labels:
ax.xaxis.set_ticklabels(['No','Yes'])
ax.xaxis.set_ticklabels(['No','Yes'])
plt.show()

from sklearn.metrics import classification_report
print(classification_report(y_test, y_prediction))