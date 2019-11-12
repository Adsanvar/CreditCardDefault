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

data = pd.read_excel('default_of_credit_card_clients.xls', header= None)

#format table
data = data.drop([0]) #drops description names
data.columns = data.iloc[0]
data = data.drop([1])
data.drop(data.columns[0], 1, inplace=True) #drops ID

#=============Data Imbalance===============#
#default = data[data.Y==1]
#n_default = data[data.Y==0]
default = data[data['default payment next month'] == 1]
n_default = data[data['default payment next month'] == 0]
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


#=============END Data Filtering===============#


# plt.scatter(n_default.X5, n_default.X1, color = 'g', alpha=.2, label ="non default")
# plt.scatter(default.X5, default.X1, color ='b', alpha=.3, label="default")


# data = np.array(data)

# selector = math.ceil((1/4)*data.shape[0])
# end = data.shape[0] - selector
# print(selector)

# test_data_features = [i for i in data[end:, :-1]]
# test_data_features_class = [i for i in data[end:, 23]]
# train_data_features = [i for i in data[:end, :-1]] # content only, no result values
# train_data_class = [i for i in data[:end, 23]]


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