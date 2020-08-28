from sklearn.decomposition import PCA
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

model = PCA(n_components = 2)
sc = StandardScaler()

dataset = load_breast_cancer()

x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.3, random_state=40)
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

x_train = model.fit_transform(x_train)
x_test = model.transform(x_test)

clf = SVC(kernel = 'linear')
clf.fit(x_train, y_train)

print("Evaluation: " + str(clf.score(x_test, y_test)))
