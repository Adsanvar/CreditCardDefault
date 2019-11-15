# import sys

# i = 2
# print ("the script has the name %s" % (sys.argv[2]))

#====================================
#OLD


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

# from sklearn.neural_network import MLPClassifier

# nn = MLPClassifier()

# nn.fit(train_data_features, train_data_class)

# print(nn.score(test_data_features, test_data_features_class))
# # print("Normalized")
# # nn.fit(norm_data_features, data_class)

# # print(nn.score(norm_test, test_class))




# #=======================