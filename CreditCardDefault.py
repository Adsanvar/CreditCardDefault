import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing

data = pd.read_excel('default_of_credit_card_clients.xls')
data = np.array(data)

#test_data_features = [data[25001:, 1:-1]] # result missing for testing purposes // results in issue when prediction...
test_data_features = data[25001:, 1:-1]
#test_data_features_features = []
test_data_features_class = data[25001:, 24]
print(test_data_features_class[0])
train_data_features = [i for i in data[1:25001, 1:-1]] # content only, no result values
train_data_class = [i for i in data[1:25001, 24]]
print(train_data_class[0])
#core_data = data[1:,1:-1] #trims data to not include headers and include limit_bal -> Pay_Amt
#df = pd.DataFrame(data) #back to dataframe for csv conversion

#print(train_data_features[0])
#print("Normalization:\n")
#print(norm_feat[0])

#non normalized
knn = KNeighborsClassifier(5)
knn.fit(train_data_features, train_data_class)

predictions = knn.predict(test_data_features)
correct =0
for x in range(0, len(test_data_features_class)):
    if(test_data_features_class[x] == predictions[x]):
        correct += 1

print(knn.score(train_data_features, train_data_class))
print("Actual Precentage, {}".format((correct/len(test_data_features_class)) * 100))

# Normalized
norm_feat = preprocessing.normalize(train_data_features)
norm_test = preprocessing.normalize(test_data_features)

knnNorm = KNeighborsClassifier(5)
knnNorm.fit(norm_feat, train_data_class)

normPredictions =  knnNorm.predict(norm_test)
norCorrect = 0

for p in range(0, len(norm_test)):
    if(test_data_features_class[p] == normPredictions[p]):
        norCorrect += 1

print(knnNorm.score(norm_feat, train_data_class))
print("Norm Actual Precentage, {}".format((norCorrect/len(test_data_features_class)) * 100))


#df.to_csv('predictions.csv')
#print("Normal")
#print(knn.score(train_data_features, train_data_class))
#print("Normalized")
#print(knnNorm.score(norm_feat, train_data_class))