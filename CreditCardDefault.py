import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Perceptron
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
#import pydot
import math
import matplotlib.pyplot as plt
from scipy.stats import norm


data = pd.read_excel('default_of_credit_card_clients.xls', header= None)

#format table
data = data.drop([0]) #drops description names
data.columns = data.iloc[0]
data = data.drop([1])
data.drop(data.columns[0], 1, inplace=True) #drops ID

#=============Data Imbalance===============#
#default = data[data.Y==1]
#n_default = data[data.Y==0]
#default = data[data['default payment next month'] == 1]
#n_default = data[data['default payment next month'] == 0]
# objects = ('default', 'non_default')
# y_pos = [0, 1]
# fig = plt.figure()
# ax = fig.add_subplot()
# ax.bar(0, len(default), align ='center', alpha=.4, label ="Default: " + str(len(default)) )
# ax.bar(1, len(n_default), align='center', color ='g', alpha =.4, label = "Non-Default: " + str(len(n_default)))
# fig.legend()
# plt.xticks(y_pos, objects)
# #fig.savefig('imbalanced_data.png', format='png')
# plt.show()

#=============END Data Imbalance===============#

#=============Data Filtering===============#
#Summary to markdown
#interest = ['SEX', 'EDUCATION', 'AGE', 'MARRIAGE', 'PAY_0','PAY_2', 'PAY_3', 'PAY_4', 'PAY_5','PAY_6']
# for col in interest:
#         f = open('report.md', 'a')
#         d = pd.DataFrame(data[col].value_counts())
#         body =""
#         head = "\n|"+col+"| |\n"# |\n|---|---|---|---|\n| | | |\n\n"
#         brk ="|---|---|\n"
#         for i in d.index:
#             #print(d.loc[i][0])
#             body += "|" +str(i)+"|"+str(d.loc[i][0]) +"|\n"

                  
#         body += "\n"
#         f.write(head)
#         f.write(brk)
#         f.write(body)
#         f.close()


##Unrepresenative data 0's all across the board For Report
# zeros = data[(data.BILL_AMT1 == 0) & (data.BILL_AMT2 == 0) & (data.BILL_AMT3 == 0) & (data.BILL_AMT4 == 0) & (data.BILL_AMT5 == 0) & (data.BILL_AMT6 == 0) & (data.PAY_AMT1 == 0) & (data.PAY_AMT2 == 0) & (data.PAY_AMT3 == 0) & (data.PAY_AMT4 == 0) & (data.PAY_AMT5 == 0) & (data.PAY_AMT6 == 0) ]
# objects = ('default', 'non_default')
# y_pos = [0, 1]
# fig = plt.figure()
# ax = fig.add_subplot()
# ax.set(title="Data of Pay -2 and 0's for BILL_AMT / PAY_AMT")
# ax.bar(0, len(zeros[zeros['default payment next month'] == 1]), align ='center', alpha=.4, label ="Default: " + str(len(zeros[zeros['default payment next month'] == 1])) +"\nRatio: " +str(len(zeros[zeros['default payment next month'] == 1])/len(default)) )
# ax.bar(1, len(zeros[zeros['default payment next month'] == 0]), align='center', color ='g', alpha =.4, label = "Non-Default: " + str(len(zeros[zeros['default payment next month'] == 0])) +"\nRatio: " +str(len(zeros[zeros['default payment next month'] == 0])/len(n_default)) )
# plt.legend()
# plt.xticks(y_pos, objects)
# fig.savefig('all_zeros.png', format='png')
#plt.show()

##This drops the data
zeros = data[(data.BILL_AMT1 == 0) & (data.BILL_AMT2 == 0) & (data.BILL_AMT3 == 0) & (data.BILL_AMT4 == 0) & (data.BILL_AMT5 == 0) & (data.BILL_AMT6 == 0) & (data.PAY_AMT1 == 0) & (data.PAY_AMT2 == 0) & (data.PAY_AMT3 == 0) & (data.PAY_AMT4 == 0) & (data.PAY_AMT5 == 0) & (data.PAY_AMT6 == 0) ].index
data.drop(zeros, inplace = True)
#data.to_csv("filtered.csv")

##Dropping 5,6,0  for Education
##For report 
# d = data.EDUCATION.value_counts()
# d2 = d.tolist()
# objects = d.index.tolist()
# y_pos = [0, 1, 2, 3, 4, 5, 6]
# colors = ['g','b','y','purple','pink','gray']
# label = str(objects[0])+": "+str(d2[0]) +"\n"+ str(objects[1])+": "+str(d2[1]) +"\n"+str(objects[2])+": "+str(d2[2]) +"\n"+str(objects[3])+": "+str(d2[3]) +"\n"+str(objects[4])+": "+str(d2[4]) +"\n"+str(objects[5])+": "+str(d2[5]) +"\n"+str(objects[6])+": "+str(d2[6])
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.set(title="Data of Education", xlabel = 'Education Category', ylabel='Number of People')
# ax.bar(y_pos, d2, align ='center', color =colors, label = label)
# #ax.bar(1, len(zeros[zeros['default payment next month'] == 0]), align='center', color ='g', alpha =.4, label = "Non-Default: " + str(len(zeros[zeros['default payment next month'] == 0])) +"\nRatio: " +str(len(zeros[zeros['default payment next month'] == 0])/len(n_default)) )
# plt.legend()
# plt.xticks(y_pos, objects)
#fig.savefig('education.png', format='png')
#plt.show()

edu = data[(data.EDUCATION == 5) | (data.EDUCATION == 6) | (data.EDUCATION == 0)].index
data.drop(edu, inplace = True)
# d = data.EDUCATION.value_counts()
# d2 = d.tolist()
# objects = d.index.tolist()
# y_pos = [0, 1, 2, 3]
# colors = ['g','b','y','purple']
# label = str(objects[0])+": "+str(d2[0]) +"\n"+ str(objects[1])+": "+str(d2[1]) +"\n"+str(objects[2])+": "+str(d2[2]) +"\n"+str(objects[3])+": "+str(d2[3])
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.set(title="Data of Education", xlabel = 'Education Category', ylabel='Number of People')
# ax.bar(y_pos, d2, align ='center', color =colors, label = label)
# #ax.bar(1, len(zeros[zeros['default payment next month'] == 0]), align='center', color ='g', alpha =.4, label = "Non-Default: " + str(len(zeros[zeros['default payment next month'] == 0])) +"\nRatio: " +str(len(zeros[zeros['default payment next month'] == 0])/len(n_default)) )
# plt.legend()
# plt.xticks(y_pos, objects)
# fig.savefig('education_2.png', format='png')

##Dropping Marriage category 0

mrg = data[(data.MARRIAGE == 0)].index
data.drop(mrg, inplace = True)

# d = data.MARRIAGE.value_counts()
# d2 = d.tolist()
# objects = d.index.tolist()
# y_pos = [0, 1, 2]
# colors = ['g','b','y']
# label = str(objects[0])+": "+str(d2[0]) +"\n"+ str(objects[1])+": "+str(d2[1]) +"\n"+str(objects[2])+": "+str(d2[2])
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.set(title="Data of Marriage", xlabel = 'Marriage Category', ylabel='Number of People')
# ax.bar(y_pos, d2, align ='center', color =colors, label = label)
# plt.legend()
# plt.xticks(y_pos, objects)
# fig.savefig('marriage_2.png', format='png')


##Dropping AGE under 18
age = data[(data.AGE < 18)].index
data.drop(age, inplace =True)

##Data Imbalanced Pt. 2 Filtered Data
# f_default = data[data['default payment next month'] == 1] #ON filtered
# f_n_default = data[data['default payment next month'] == 0]
# objects = ('default', 'non_default')
# y_pos = [0, 1]
# fig = plt.figure()
# ax = fig.add_subplot()
# ax.set(title='Filtered Imbalanced')
# ax.bar(0, len(f_default), align ='center', alpha=.4, label ="Default: " + str(len(f_default)) )
# ax.bar(1, len(f_n_default), align='center', color ='g', alpha =.4, label = "Non-Default: " + str(len(f_n_default)))
# fig.legend()
# plt.xticks(y_pos, objects)
# fig.savefig('filtered_imbalanced_data.png', format='png')
#plt.show()

##Histogram of Age Vs Limit Balance
#objects = data.AGE.tolist() #'names of each bar on x-axis
#y_pos = [0, 1, 2]
# label = str(objects[0])+": "+str(d2[0]) +"\n"+ str(objects[1])+": "+str(d2[1]) +"\n"+str(objects[2])+": "+str(d2[2])
#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.set(title="AGE VS LIMIT_BAL", xlabel = 'AGE', ylabel='LIMIT_BAL')
#ax.bar(objects, data.LIMIT_BAL.tolist(), align ='center')
# plt.legend()
#plt.xticks(y_pos, objects)
#fig.savefig('skewed_right_filtered_Data.png', format='png')
#plt.show()

##### MODIFIED BELOW
# # ##Normal Curve - pre dropping outliers
# # # mean = np.mean(objects)
# # # median = np.median(objects)
# # # var = np.var(objects)
# # # sd = math.sqrt(var)
# # # ub = np.max(objects)
# # # lb = np.min(objects)
# # # q1 = np.quantile(objects, .25)
# # # q3 = np.quantile(objects, .75)
# # # IQR = q3-q1
# # # lower_outliers = math.floor(q1 - (1.5*IQR))
# # # upper_outliers = math.floor(q3 + (1.5*IQR))

# # # x = np.arange(lb,ub,1) #used for 'normal'
# # # plt.style.use('fivethirtyeight')
# # # ax.plot(x, norm.pdf(x, mean,sd)) #used for 'normal' Curve
# # # y = norm.pdf(x,mean,sd) #used for 'normal'
# # # xq3 = np.arange(q3, ub,1)
# # # yq3 = norm.pdf(xq3, mean, sd)
# # # xq1 = np.arange(lb, q1,1)
# # # yq1 = norm.pdf(xq1, mean, sd)
# # # ax.fill_between(xq3,yq3,0, alpha=.3, color='r', label="Q3: " +str(q3) +"\nOutliers > " +str(upper_outliers)+" :: QTY: " +str(len(data[(data.AGE >60)])))
# # # ax.fill_between(xq1,yq1,0, alpha=.3, color='g', label="Q1: " +str(q1) +"\nOutliers < " +str(lower_outliers)+" :: QTY: " + str(len(data[(data.AGE) < lower_outliers])))
# # # ax.set_title('Skewed Normal Gaussian Curve')
# # # plt.legend()
# # # fig.savefig('Skewed_Normal_Gaussian_Cruve.png', format='png')
# # # #plt.show()

# # ##Removing Outliers
# # ####outliers = data[(data.AGE >60)].index
# # ####data.drop(outliers, inplace=True)

# # #new_objects = data.AGE.tolist() #'names of each bar on x-axis
# # # #y_pos = [0, 1, 2]
# # # # label = str(objects[0])+": "+str(d2[0]) +"\n"+ str(objects[1])+": "+str(d2[1]) +"\n"+str(objects[2])+": "+str(d2[2])
# # # ax.set(title="AGE VS LIMIT_BAL", xlabel = 'AGE', ylabel='LIMIT_BAL')
# # # ax.bar(new_objects, data.LIMIT_BAL.tolist(), align ='center')

# # # #plt.legend()
# # # fig.savefig('no_outliers_filtered_Data.png', format='png')
# # # #plt.show()

# # ##Normal Curve With no outliers
# # # mean = np.mean(new_objects)
# # # median = np.median(new_objects)
# # # var = np.var(new_objects)
# # # sd = math.sqrt(var)
# # # ub = np.max(new_objects)
# # # lb = np.min(new_objects)
# # # q1 = np.quantile(new_objects, .25)
# # # q3 = np.quantile(new_objects, .75)
# # # IQR = q3-q1
# # # lower_outliers = math.floor(q1 - (1.5*IQR))
# # # upper_outliers = math.floor(q3 + (1.5*IQR))

# # # x = np.arange(lb,ub,1) 
# # # plt.style.use('fivethirtyeight')
# # # ax.plot(x, norm.pdf(x, mean,sd)) 
# # # y = norm.pdf(x,mean,sd) 
# # # xq3 = np.arange(q3, ub,1)
# # # yq3 = norm.pdf(xq3, mean, sd)
# # # xq1 = np.arange(lb, q1,1)
# # # yq1 = norm.pdf(xq1, mean, sd)
# # # ax.fill_between(xq3,yq3,0, alpha=.3, color='r', label="Q3: " +str(q3) )
# # # ax.fill_between(xq1,yq1,0, alpha=.3, color='g', label="Q1: " +str(q1) )
# # # ax.set_title('Normal Gaussian Curve')
# # # plt.legend()
# # # fig.savefig('No_Outliers_Skewed_Normal_Gaussian_Cruve.png', format='png')
# # # #plt.show()

# # ##Resulting Unbalanced
# # # final_default = data[data['default payment next month'] == 1] #ON filtered
# # # final_n_default = data[data['default payment next month'] == 0]
# # # objects = ('default', 'non_default')
# # # y_pos = [0, 1]
# # # fig = plt.figure()
# # # ax = fig.add_subplot(111)
# # # ax.set(title='Filtered Imbalanced')
# # # ax.bar(0, len(final_default), align ='center', alpha=.4, label ="Default: " + str(len(final_default)) )
# # # ax.bar(1, len(final_n_default), align='center', color ='g', alpha =.4, label = "Non-Default: " + str(len(final_n_default)))
# # # fig.legend()
# # # plt.xticks(y_pos, objects)
# # # fig.savefig('final_imbalanced_data.png', format='png')
# # #plt.show()

#=============END Data Filtering===============#

#============DATA SELECTION - DATA ANALYSING==================#

selector = math.ceil((1/6)*data.shape[0]) #percentage of data to be obtained
end = data.shape[0] - selector #the point at which it is

#test_data = data.loc[end+1:, :]
train_data = data.loc[:end, :]

default = train_data[train_data['default payment next month'] == 1]
n_default = train_data[train_data['default payment next month'] == 0]

train_data_features = data.loc[:end, :'default payment next month']
train_data_class = data.loc[:end, 'default payment next month']

test_data_features = data.loc[end+1:, :'default payment next month']
test_data_class = data.loc[end+1:, 'default payment next month']


fig = plt.figure()
ax = fig.add_subplot(111)
def_objects = default.AGE.tolist()
n_def_objects = n_default.AGE.tolist()
l_def_objects = default.LIMIT_BAL.tolist()
l_n_def_objects = n_default.LIMIT_BAL.tolist()
e_def_objects = default.EDUCATION.tolist()
e_n_def_objects = n_default.EDUCATION.tolist()

##def_objects Values
mean = np.mean(def_objects)
median = np.median(def_objects)
var = np.var(def_objects)
sd = math.sqrt(var)
ub = np.max(def_objects)
lb = np.min(def_objects)
q1 = np.quantile(def_objects, .25)
q3 = np.quantile(def_objects, .75)
IQR = q3-q1
lower_outliers = math.floor(q1 - (1.5*IQR))
upper_outliers = math.floor(q3 + (1.5*IQR))

outliers = default[(default.AGE > upper_outliers)].index
default.drop(outliers, inplace = True)

# x = np.arange(lb,ub,1) 
# plt.style.use('fivethirtyeight')
# ax.plot(x, norm.pdf(x, mean,sd) ,label="Default Class") 
# y = norm.pdf(x,mean,sd) 
# xq3 = np.arange(q3, ub,1)
# yq3 = norm.pdf(xq3, mean, sd)
# xq1 = np.arange(lb, q1,1)
# yq1 = norm.pdf(xq1, mean, sd)
# ax.fill_between(xq3,yq3,0, alpha=.3, color='r', label="Q3: " +str(q3) +"\nOutliers > " +str(upper_outliers)+" :: QTY: " +str(len(default[(default.AGE >upper_outliers)])))
# ax.fill_between(xq1,yq1,0, alpha=.3, color='g', label="Q1: " +str(q1) +"\nOutliers < " +str(lower_outliers)+" :: QTY: " + str(len(default[(default.AGE) < lower_outliers])))
# plt.legend()
# fig.savefig('default_Skewed_Normal_Gaussian_Cruve.png', format='png')

##n_def_objects Values
mean = np.mean(n_def_objects)
median = np.median(n_def_objects)
var = np.var(n_def_objects)
sd = math.sqrt(var)
ub = np.max(n_def_objects)
lb = np.min(n_def_objects)
q1 = np.quantile(n_def_objects, .25)
q3 = np.quantile(n_def_objects, .75)
IQR = q3-q1
lower_outliers = math.floor(q1 - (1.5*IQR))
upper_outliers = math.floor(q3 + (1.5*IQR))

outliers = n_default[(n_default.AGE > upper_outliers)].index
n_default.drop(outliers, inplace = True)

# x = np.arange(lb,ub,1) 
# plt.style.use('fivethirtyeight')
# ax.plot(x, norm.pdf(x, mean,sd), color = 'orange', label="Non-Default Class") 
# y = norm.pdf(x,mean,sd) 
# xq3 = np.arange(q3, ub,1)
# yq3 = norm.pdf(xq3, mean, sd)
# xq1 = np.arange(lb, q1,1)
# yq1 = norm.pdf(xq1, mean, sd)
# ax.fill_between(xq3,yq3,0, alpha=.4, color='y', label="Q3: " +str(q3) +"\nOutliers > " +str(upper_outliers)+" :: QTY: " +str(len(n_default[(n_default.AGE >60)])))
# ax.fill_between(xq1,yq1,0, alpha=.4, color='b', label="Q1: " +str(q1) +"\nOutliers < " +str(lower_outliers)+" :: QTY: " + str(len(n_default[(n_default.AGE) < lower_outliers])))

# ax.set_title('Normal Gaussian Curve')
# plt.legend()
# fig.savefig('non_default_Skewed_Normal_Gaussian_Cruve.png', format='png')
#plt.show()


###For Limit_Balance - NON_DEFAULT
mean = np.mean(l_n_def_objects)
median = np.median(l_n_def_objects)
var = np.var(l_n_def_objects)
sd = math.sqrt(var)
ub = np.max(l_n_def_objects)
lb = np.min(l_n_def_objects)
q1 = np.quantile(l_n_def_objects, .25)
q3 = np.quantile(l_n_def_objects, .75)
IQR = q3-q1
lower_outliers = math.floor(q1 - (1.5*IQR))
upper_outliers = math.floor(q3 + (1.5*IQR))

outliers = n_default[(n_default.LIMIT_BAL > upper_outliers)].index
n_default.drop(outliers, inplace = True)

# x = np.arange(lb,ub,1) 
# plt.style.use('fivethirtyeight')
# ax.plot(x, norm.pdf(x, mean,sd), color = 'orange', label="Non-Default Class") 
# y = norm.pdf(x,mean,sd) 
# xq3 = np.arange(q3, ub,1)
# yq3 = norm.pdf(xq3, mean, sd)
# xq1 = np.arange(lb, q1,1)
# yq1 = norm.pdf(xq1, mean, sd)
# ax.fill_between(xq3,yq3,0, alpha=.4, color='y', label="Q3: " +str(q3) +"\nOutliers > " +str(upper_outliers)+" :: QTY: " +str(len(n_default[(n_default.LIMIT_BAL > upper_outliers)])))
# ax.fill_between(xq1,yq1,0, alpha=.4, color='b', label="Q1: " +str(q1) +"\nOutliers < " +str(lower_outliers)+" :: QTY: " + str(len(n_default[(n_default.LIMIT_BAL < lower_outliers)])))

# ax.set_title('Normal Gaussian Curve')
# plt.legend()
# fig.savefig('non_default_Skewed_Normal_Gaussian_Cruve_LIMIT_BAL.png', format='png')

##LIMIT_BAL DEFAULT
mean = np.mean(l_def_objects)
median = np.median(l_def_objects)
var = np.var(l_def_objects)
sd = math.sqrt(var)
ub = np.max(l_def_objects)
lb = np.min(l_def_objects)
q1 = np.quantile(l_def_objects, .25)
q3 = np.quantile(l_def_objects, .75)
IQR = q3-q1
lower_outliers = math.floor(q1 - (1.5*IQR))
upper_outliers = math.floor(q3 + (1.5*IQR))

outliers = default[(default.LIMIT_BAL > upper_outliers)].index
default.drop(outliers, inplace = True)
# x = np.arange(lb,ub,1) 
# plt.style.use('fivethirtyeight')
# ax.plot(x, norm.pdf(x, mean,sd), label="Default Class") 
# y = norm.pdf(x,mean,sd) 
# xq3 = np.arange(q3, ub,1)
# yq3 = norm.pdf(xq3, mean, sd)
# xq1 = np.arange(lb, q1,1)
# yq1 = norm.pdf(xq1, mean, sd)
# ax.fill_between(xq3,yq3,0, alpha=.4, color='y', label="Q3: " +str(q3) +"\nOutliers > " +str(upper_outliers)+" :: QTY: " +str(len(default[(default.LIMIT_BAL > upper_outliers)])))
# ax.fill_between(xq1,yq1,0, alpha=.4, color='b', label="Q1: " +str(q1) +"\nOutliers < " +str(lower_outliers)+" :: QTY: " + str(len(default[(default.LIMIT_BAL < lower_outliers)])))

# ax.set_title('Normal Gaussian Curve')
# # plt.legend()
# fig.savefig('default_Skewed_Normal_Gaussian_Cruve_LIMIT_BAL.png', format='png')

##For EDUCATION
mean = np.mean(e_n_def_objects)
median = np.median(e_n_def_objects)
var = np.var(e_n_def_objects)
sd = math.sqrt(var)
ub = np.max(e_n_def_objects)
lb = np.min(e_n_def_objects)
q1 = np.quantile(e_n_def_objects, .25)
q3 = np.quantile(e_n_def_objects, .75)
IQR = q3-q1
lower_outliers = math.floor(q1 - (1.5*IQR))
upper_outliers = math.floor(q3 + (1.5*IQR))

# outliers = n_default[(n_default.EDUCATION > upper_outliers)].index
# n_default.drop(outliers, inplace = True)

# x = np.arange(lb,ub,1) 
# plt.style.use('fivethirtyeight')
# ax.plot(x, norm.pdf(x, mean,sd), color = 'orange', label="Non-Default Class") 
# y = norm.pdf(x,mean,sd) 
# xq3 = np.arange(q3, ub,1)
# yq3 = norm.pdf(xq3, mean, sd)
# xq1 = np.arange(lb, q1,1)
# yq1 = norm.pdf(xq1, mean, sd)
# ax.fill_between(xq3,yq3,0, alpha=.4, color='y', label="Q3: " +str(q3) +"\nOutliers > " +str(upper_outliers)+" :: QTY: " +str(len(n_default[(n_default.EDUCATION > upper_outliers)])))
# ax.fill_between(xq1,yq1,0, alpha=.4, color='b', label="Q1: " +str(q1) +"\nOutliers < " +str(lower_outliers)+" :: QTY: " + str(len(n_default[(n_default.EDUCATION < lower_outliers)])))

# ax.set_title('Normal Gaussian Curve')
# plt.legend()
# fig.savefig('non_default_Skewed_Normal_Gaussian_Cruve_EDUCATION.png', format='png')

##EDUCATION DEFAULT
mean = np.mean(e_def_objects)
median = np.median(e_def_objects)
var = np.var(e_def_objects)
sd = math.sqrt(var)
ub = np.max(e_def_objects)
lb = np.min(e_def_objects)
q1 = np.quantile(e_def_objects, .25)
q3 = np.quantile(e_def_objects, .75)
IQR = q3-q1
lower_outliers = math.floor(q1 - (1.5*IQR))
upper_outliers = math.floor(q3 + (1.5*IQR))

# outliers = default[(default.EDUCATION > upper_outliers)].index
# default.drop(outliers, inplace = True)

# x = np.arange(lb,ub,1) 
# plt.style.use('fivethirtyeight')
# ax.plot(x, norm.pdf(x, mean,sd), label="Default Class") 
# y = norm.pdf(x,mean,sd) 
# xq3 = np.arange(q3, ub,1)
# yq3 = norm.pdf(xq3, mean, sd)
# xq1 = np.arange(lb, q1,1)
# yq1 = norm.pdf(xq1, mean, sd)
# ax.fill_between(xq3,yq3,0, alpha=.4, color='y', label="Q3: " +str(q3) +"\nOutliers > " +str(upper_outliers)+" :: QTY: " +str(len(default[(default.EDUCATION > upper_outliers)])))
# ax.fill_between(xq1,yq1,0, alpha=.4, color='b', label="Q1: " +str(q1) +"\nOutliers < " +str(lower_outliers)+" :: QTY: " + str(len(default[(default.EDUCATION < lower_outliers)])))

# ax.set_title('Normal Gaussian Curve')
# plt.legend()
# fig.savefig('default_Skewed_Normal_Gaussian_Cruve_EDUCATION.png', format='png')

# #FINAL IMbalanced
# objects = ('default', 'non_default')
# y_pos = [0, 1]

# ax.set(title='Filtered Imbalanced')
# ax.bar(0, len(default), align ='center', alpha=.4, label ="Default: " + str(len(default)) )
# ax.bar(1, len(n_default), align='center', color ='g', alpha =.4, label = "Non-Default: " + str(len(n_default)))
# fig.legend()
# plt.xticks(y_pos, objects)
# fig.savefig('final_imbalanced_data.png', format='png')

# from mpl_toolkits import mplot3d
# ax = plt.axes(projection = '3d')
# ax.scatter(default.AGE, default.LIMIT_BAL, default.EDUCATION, label ='Default')
# ax.scatter(n_default.AGE, n_default.LIMIT_BAL, n_default.EDUCATION, label='Non_Default')
# ax.set(title='AGE VS LIMIT_BAL VS EDUCATION', xlabel='AGE', zlabel='EDUCATION', ylabel='LIMIT_BAL')
# plt.legend()
# #Default
# # ax.scatter(default.AGE, default.LIMIT_BAL)
# # ax.scatter(n_default.AGE, n_default.LIMIT_BAL) 
# fig.savefig('scatter_outliers.png', format='png')
# #plt.show()

##====================STANDARDIZED
#selecting only from train dataset
selection = pd.concat([train_data.LIMIT_BAL, train_data.EDUCATION, train_data.AGE, train_data['default payment next month']], axis = 1)

selection_default =selection[selection['default payment next month'] == 1]
selection_default_class = selection_default.loc[:, 'default payment next month']
selection_default.drop('default payment next month', axis = 1,inplace = True)

selection_n_default =selection[selection['default payment next month'] == 0]
selection_n_default_class = selection_n_default.loc[:, 'default payment next month']
selection_n_default.drop('default payment next month', axis = 1,inplace = True)

stand_default = StandardScaler().fit_transform(selection_default)
stand_n_default = StandardScaler().fit_transform(selection_n_default)

stand_def_objects = stand_default.tolist()
stand_n_def_objects = stand_n_default.tolist()

##standardize def_objects Values
# mean = np.mean(stand_def_objects)
# median = np.median(stand_def_objects)
# var = np.var(stand_def_objects)
# sd = math.sqrt(var)
# ub = np.max(stand_def_objects)
# lb = np.min(stand_def_objects)
# q1 = np.quantile(stand_def_objects, .25)
# q3 = np.quantile(stand_def_objects, .75)
# IQR = q3-q1
# lower_outliers = math.floor(q1 - (1.5*IQR))
# upper_outliers = math.floor(q3 + (1.5*IQR))

# x = np.arange(lb,ub,1) 
# plt.style.use('fivethirtyeight')
# ax.plot(x, norm.pdf(x, mean,sd) ,label="Default Class")
# ax.set(title="STANDARDIZED DATA")
# y = norm.pdf(x,mean,sd) 
# xq3 = np.arange(q3, ub,1)
# yq3 = norm.pdf(xq3, mean, sd)
# xq1 = np.arange(lb, q1,1)
# yq1 = norm.pdf(xq1, mean, sd)
# ax.fill_between(xq3,yq3,0, alpha=.3, color='r', label="Q3: " +str(q3) +"\nOutliers > " +str(upper_outliers)+" :: QTY: " +str(len(stand_default[(stand_default >upper_outliers)])))
# ax.fill_between(xq1,yq1,0, alpha=.3, color='g', label="Q1: " +str(q1) +"\nOutliers < " +str(lower_outliers)+" :: QTY: " + str(len(stand_default[(stand_default < lower_outliers)])))
# plt.legend()
# plt.show()
#fig.savefig('default_Skewed_Normal_Gaussian_Cruve.png', format='png')

##standardize n_def_objects Values
# ax = fig.add_subplot(111)
# mean = np.mean(stand_n_def_objects)
# median = np.median(stand_n_def_objects)
# var = np.var(stand_n_def_objects)
# sd = math.sqrt(var)
# ub = np.max(stand_n_def_objects)
# lb = np.min(stand_n_def_objects)
# q1 = np.quantile(stand_n_def_objects, .25)
# q3 = np.quantile(stand_n_def_objects, .75)
# IQR = q3-q1
# lower_outliers = math.floor(q1 - (1.5*IQR))
# upper_outliers = math.floor(q3 + (1.5*IQR))

# x = np.arange(lb,ub,1) 
# plt.style.use('fivethirtyeight')
# ax.plot(x, norm.pdf(x, mean,sd) ,label="Non Defualt Class")
# ax.set(title="STANDARDIZED DATA")
# y = norm.pdf(x,mean,sd) 
# xq3 = np.arange(q3, ub,1)
# yq3 = norm.pdf(xq3, mean, sd)
# xq1 = np.arange(lb, q1,1)
# yq1 = norm.pdf(xq1, mean, sd)
# ax.fill_between(xq3,yq3,0, alpha=.3, color='r', label="Q3: " +str(q3) +"\nOutliers > " +str(upper_outliers)+" :: QTY: " +str(len(stand_n_default[(stand_n_default > upper_outliers)])))
# ax.fill_between(xq1,yq1,0, alpha=.3, color='g', label="Q1: " +str(q1) +"\nOutliers < " +str(lower_outliers)+" :: QTY: " + str(len(stand_n_default[(stand_n_default < lower_outliers)])))
# plt.legend()
# plt.show()

ax.scatter(stand_default[:, 0], stand_default[:, 1], stand_default[:, 2])
from mpl_toolkits import mplot3d
ax = plt.axes(projection = '3d')
ax.scatter(stand_default[:, 0], stand_default[:, 2], stand_default[:, 1], label ='Default')
ax.scatter(stand_n_default[:, 0], stand_n_default[:, 2], stand_n_default[:, 1],  label='Non_Default')
ax.set(title='Standardized Data', xlabel='LIMIT_BAL', zlabel='AGE', ylabel='EDUCATION')
plt.legend()
#Default
# ax.scatter(default.AGE, default.LIMIT_BAL)
# ax.scatter(n_default.AGE, n_default.LIMIT_BAL) 
#fig.savefig('scatter_outliers.png', format='png')
plt.show()


#============END DATA SELCETION - DATA ANALYSING==================#


# plt.scatter(n_default.X5, n_default.X1, color = 'g', alpha=.2, label ="non default")
# plt.scatter(default.X5, default.X1, color ='b', alpha=.3, label="default")


# #print(train_data_features)
# #Normalized Data
# norm_test = StandardScaler().fit_transform(test_data_features)
# norm_train = StandardScaler().fit_transform(train_data_features)


# #=======================
# print("Non normalized Best K = 42")
# #134 - > 79%
# #42 -> 79%
# #non normalized
# knn = KNeighborsClassifier(42)

# knn.fit(train_data_features, train_data_class)

# predictions = knn.predict(test_data_features)
# correct =0
# for x in range(0, len(test_data_features_class)):
#     if(test_data_features_class[x] == predictions[x]):
#         correct += 1

# print(knn.score(test_data_features, test_data_features_class))
# print("Actual Precentage, {}".format((correct/len(test_data_features_class)) * 100))


# #KFold To check the best possible parameter
# #===============================
# #grid = {'n_neighbors': np.arange(1,100)}
# # grid = {'n_neighbors': np.arange(1, 200)}
# # #knn_search = GridSearchCV(knn, grid, cv=100)
# # knn_search = GridSearchCV(knn, grid, cv=5)
# # X = [i for i in data[1:, 1:-1]]
# # y = [i for i in data[1:, 24]]
# # knn_search.fit(X ,y )
# # print(knn_search.best_params_)
# #print(knn_search.best_score_)

# #===============================
# # best = 0
# # at = 0

# # for i in range(1, 5000):
# #     knn = KNeighborsClassifier(i)
# #     knn.fit(train_data_features, train_data_class)
# #     predictions = knn.predict(test_data_features)
# #     score = knn.score(test_data_features, test_data_features_class)
# #     print(i)
# #     if score > best:
# #         best = score
# #         at = i
# #         print("The Best Score is: " + str(best) + " at: " + str(at))

# #=======================



# #=======================

# print("Normalized Data Best K = 10")
# # 10 -> .82

# knn = KNeighborsClassifier(10)

# knn.fit(norm_train, train_data_class)

# predictions = knn.predict(norm_test)

# print(knn.score(norm_test, test_data_features_class))

# #=======================
# # best = 0
# # at = 0

# # for i in range(1, 5000):
# #     knn = KNeighborsClassifier(i)
# #     knn.fit(norm_train, train_data_class)
# #     predictions = knn.predict(norm_test)
# #     score = knn.score(norm_test, test_data_features_class)
# #     print(i)
# #     if score > best:
# #         best = score
# #         at = i
# #         print("The Best Score is: " + str(best) + " at: " + str(at))

# #=======================

# #=======================




# #=======================
# print("Normalized with PCA Best K = 11, n_components = 3")
# #11 -> 80%
# #covaraince
# c = np.cov(norm_test.T)
# #eigen values / vectors from data
# w, v = np.linalg.eig(c)
# pca = PCA(n_components = 3)
# comp_test = pca.fit_transform(norm_test)

# #covaraince
# c = np.cov(norm_train.T)
# #eigen values / vectors from data
# w, v = np.linalg.eig(c)
# pca = PCA(n_components = 3)
# comp_train = pca.fit_transform(norm_train)

# # 2 = 81% at k =12
# # 3 = 81% at k =11

# knn = KNeighborsClassifier(11)

# knn.fit(comp_train, train_data_class)

# predictions = knn.predict(comp_test)

# print(knn.score(comp_test, test_data_features_class))

# #===================
# # best = 0
# # at = 0

# # for i in range(1, 5000):
# #     knn = KNeighborsClassifier(i)
# #     knn.fit(comp_train, train_data_class)
# #     predictions = knn.predict(comp_test)
# #     score = knn.score(comp_test, test_data_features_class)
# #     print(i)
# #     if score > best:
# #         best = score
# #         at = i
# #         print("The Best Score is: " + str(best) + " at: " + str(at))

# #===================


# #=======================



# #=======================
# # print("Percentron: ")

# # clf = Perceptron()

# # clf.fit(train_data_features , train_data_class)

# # print("Normal: " + str(clf.score(test_data_features, test_data_features_class)))

# # clf.fit(norm_train , train_data_class)

# # print("Standardized: " + str(clf.score(norm_test, test_data_features_class)))

# # clf.fit(comp_train , train_data_class)

# # print("PCA: " + str(clf.score(comp_test, test_data_features_class)))


# #=======================


# #=======================
# print("Random Forest: ")

# # data_features = [i for i in data[1:, 1:-1]]
# # data_class = [i for i in data[1:, 24]]
# # norm_data_features = StandardScaler().fit_transform(data_features)


# # #covaraince
# # c = np.cov(norm_data_features.T)
# # #eigen values / vectors from data
# # w, v = np.linalg.eig(c)
# # pca = PCA(n_components = 3)
# # comp_test = pca.fit_transform(norm_data_features)


# # #rfc = RandomForestClassifier(n_estimators=100,random_state=42, max_depth=3)
# # # rfc = RandomForestClassifier()
# # # rfc.fit(data_features , data_class)

# # # print("Normal: " + str(rfc.score(data_features, data_class)))

# # # rfc.fit(norm_data_features , data_class)

# # # print("Standardized: " + str(rfc.score(norm_data_features, data_class)))

# # # rfc.fit(comp_test , data_class)

# # # print("PCA: " + str(rfc.score(comp_test, data_class)))
# # rfc = RandomForestClassifier()
# # rfc.fit(data_features , data_class)

# # print("Normal: " + str(rfc.score(data_features, data_class)))

# # rfc.fit(norm_data_features , data_class)

# # print("Standardized: " + str(rfc.score(norm_data_features, data_class)))

# # rfc.fit(comp_test , data_class)

# # print("PCA: " + str(rfc.score(comp_test, data_class)))

# #=======================

# #=======================
# print("Random Forest 2: ")

# # # data_features = [i for i in data[1:29990, 1:-1]]
# # # data_class = [i for i in data[1:29990, 24]]
# # # norm_data_features = StandardScaler().fit_transform(data_features)

# # # rfc = RandomForestClassifier()
# # # rfc.fit(data_features , data_class)

# # # print("Normal: " + str(rfc.score(data_features, data_class)))

# # # rfc.fit(norm_data_features , data_class)

# # # print("Standardized: " + str(rfc.score(norm_data_features, data_class)))
# # test_feat = [i for i in data[29991:, 2:-19]]
# # test_class =[i for i in data[29991:, 24]]
# # personal_data_features = [i for i in data[1:29990, 2:-19]]
# # data_class = [i for i in data[1:29990, 24]]
# # #norm_data_features = StandardScaler().fit_transform(data_features)

# # rf = RandomForestClassifier(n_estimators=100)

# # rf.fit(personal_data_features, data_class)

# # print(rf.score(test_feat, test_class))

# # for i in range(1, 10):
# #     print(rf.predict([test_feat[1]]))

# # #print(rf.predict([test_feat[0]]))
# # print(test_class)

# #________________________


# # #covaraince
# # c = np.cov(norm_data_features.T)
# # #eigen values / vectors from data
# # w, v = np.linalg.eig(c)
# # pca = PCA(n_components = 3)
# # comp_test = pca.fit_transform(norm_data_features)


# rfc = RandomForestClassifier(n_estimators=100)
# rfc.fit(train_data_features , train_data_class)

# print("Normal: " + str(rfc.score(test_data_features, test_data_features_class)))

# rfc2 = RandomForestClassifier(n_estimators=100)
# rfc2.fit(norm_train , train_data_class)

# print("Standardized: " + str(rfc2.score(norm_test, test_data_features_class)))

# # rfc.fit(comp_test , test_data_features_class)

# # print("PCA: " + str(rfc.score(comp_test, test_data_features_class)))

# #=======================

# #=======================

# from sklearn.neural_network import MLPClassifier

# nn = MLPClassifier()

# nn.fit(train_data_features, train_data_class)

# print(nn.score(test_data_features, test_data_features_class))
# # print("Normalized")
# # nn.fit(norm_data_features, data_class)

# # print(nn.score(norm_test, test_class))




# #=======================