from sklearn.neural_network import MLPClassifier

#[height, weight, shoe size]
x = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39], [177, 70, 40], 
[159,55,35], [171, 75, 42], [181, 85, 43]]

y = ['male', 'female', 'female', 'female', 'male', 'male', 'male', 'female', 'male', 'female', 'male']

clf = MLPClassifier(solver="lbfgs", alpha=1e-5, hidden_layer_sizes=(5,2), random_state=1)

clf = clf.fit(x,y)

prediction = clf.predict([[190,70,43]])

print(prediction)