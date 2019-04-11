import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, normalize
from sklearn.model_selection import train_test_split
from sklearn import utils
from skimage.feature import greycomatrix, greycoprops
import pandas as pd
import pickle
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV



class ClassificationRandomForest:
    def __init__(self):
        self.data_path = ''

        self.classes = []

        self.labels = []
        self.data = []

        self.train_data = []
        self.train_labels = []

        self.test_data = []
        self.test_labels = []

        self.classification = None

        self.predicted_labels = []

        self.confusion_matrix = np.zeros((6, 6), dtype=int)
        self.normalized_confusion_matrix = np.zeros((6, 6), dtype=float)

        self.total_test_samples = 0

        self.TP = []
        self.TN = []
        self.FN = []
        self.FP = []

        self.fpr = []
        self.tpr = []
        self.th = []

    def set_data_path(self, path):
        self.data_path = path

    def extract_features(self, image):
        (B, G, R) = cv2.split(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        (H, S, V) = cv2.split(image)
        return [np.mean(B), np.mean(G), np.mean(R), np.mean(H), np.mean(S), np.std(B), np.std(G), np.std(R),
                    np.std(H), np.std(S)]

    def set_images_and_labels(self):
        dir_list = os.listdir(self.data_path)
        for directory in dir_list:
            self.classes.append(directory)
            img_list = os.listdir(os.path.join(self.data_path, directory))
            for image in img_list:
                img = cv2.imread(os.path.join(self.data_path, directory, image))
                img = cv2.resize(img, (128, 128))
                self.labels.append(directory)
                self.data.append(self.extract_features(img))

    def encode_labels(self):
        le = LabelEncoder()
        self.labels = le.fit_transform(self.labels)

    def normalize_data(self):
        self.data = normalize(self.data)

    def split_data(self):
        self.train_data, self.test_data, self.train_labels, self.test_labels = train_test_split(
            self.data, self.labels, test_size=0.2, random_state=50)

    def train(self):
        # Random Forest model
        self.classification = RandomForestClassifier(n_estimators=100, criterion='entropy', max_features='log2',
                                                     max_depth=100, random_state=5, verbose=10)
        # grid = GridSearchCV(estimator=RandomForestClassifier(), param_grid={'n_estimators': [50, 100, 200, 500],
                                                  # 'criterion': ['entropy', 'gini'],
                                                  # 'max_features': ['auto', 'sqrt', 'log2', None],
                                                  # 'max_depth': [50, 100, 200, 500]}, scoring="accuracy", cv=5,
                                                  # verbose=10)

        # Training
        self.classification.fit(self.train_data, self.train_labels)
        # print(grid.best_params_)
        # feature_imp = pd.Series(cl.feature_importances_,index=["b mean", "g mean", "r mean", "b dev", "g dev",
        # "r dev"]).sort_values(ascending=False)

        # Salvare model
        pickle.dump(self.classification, open('D:/Licenta/Models/random_forest_model.sav', 'wb'))

    def get_train_accuracy(self):
        # Calcul acuratete pentru datele de antrenare
        return accuracy_score(self.train_labels, self.classification.predict(self.train_data))

    def predict(self):
        # Testing
        # model = pickle.load(open('D:/Licenta/svm_model.sav', 'rb'))
        self.predicted_labels = self.classification.predict(self.test_data)

    def get_test_accuracy(self):
        # self.cm_classes = np.union1d(self.test_labels, self.predicted_labels)
        return accuracy_score(self.test_labels, self.predicted_labels)

    def generate_confusion_matrix(self):
        self.confusion_matrix = confusion_matrix(self.test_labels, self.predicted_labels)
        return str(confusion_matrix(self.test_labels, self.predicted_labels))

    def normalize_confusion_matrix(self):
        suma = self.confusion_matrix.sum(axis=1)
        for i in range(len(self.classes)):
            for j in range(len(self.classes)):
                if suma[i] > 0:
                    self.normalized_confusion_matrix[i, j] = self.confusion_matrix.astype('float')[i, j] / suma[i]

    def plot_confusion_matrix(self):
        # Creare figura si axe
        fig, ax = plt.subplots()

        # Heatmap
        im = ax.imshow(self.normalized_confusion_matrix, cmap='summer')

        # Colorbar
        ax.figure.colorbar(im, ax=ax)
        # cbar.ax.set_ylabel("Images", rotation=-90, va="bottom")

        ax.set_xticks(np.arange(len(self.classes)))
        ax.set_yticks(np.arange(len(self.classes)))

        ax.set_xticklabels(range(len(self.classes)))
        ax.set_yticklabels(range(len(self.classes)))


        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')

        # ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

        # Completare tabel
        for i in range(len(self.classes)):
            for j in range(len(self.classes)):
                if self.normalized_confusion_matrix[i, j] < 0.5:
                    color = 'w'
                else:
                    color = 'k'
                ax.text(j, i, '{:.2f}'.format(self.normalized_confusion_matrix[i, j]), ha="center", va="center",
                        color=color)

        ax.set_title("Normalized Confusion Matrix")
        #ax.text(10,10,"dfs", transform=ax.transAxes, fontsize=8, verticalalignment='top', ha='right', boxstyle=round)
        # Ajustare automata a parametrilor figurii
        fig.tight_layout()
        fig.savefig("D:/Licenta/Interfata/resources/images/confusion_matrix_RF.png", dpi=500)
        # plt.show()


    def get_values(self, matrix, classes):
        self.total_test_samples = np.sum(matrix[:, :])

        for each in classes:
            class_num = self.classes.index(each)
            self.TP.append(matrix[class_num, class_num])
            tn = 0
            fn = 0
            fp = 0
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    if i != class_num and j != class_num:
                        tn += matrix[i, j]
                    if i == class_num and j != class_num:
                        fn += matrix[i, j]
                    if i != class_num and j == class_num:
                        fp += matrix[i, j]

            self.TN.append(tn)
            self.FN.append(fn)
            self.FP.append(fp)
        self.TP = np.array(self.TP)
        self.TN = np.array(self.TN)
        self.FP = np.array(self.FP)
        self.FN = np.array(self.FN)

    def get_test_accuracy_oonf_matr(self):
        acc = np.sum(self.TP) / self.total_test_samples
        return acc

    def get_test_recall_per_class(self):
        self.get_values(self.confusion_matrix, self.classes)
        recall = []
        for i in range(len(self.classes)):
            if (self.TP[i] + self.FN[i]) != 0:
                recall.append(self.TP[i] / (self.TP[i] + self.FN[i]))
            else:
                recall.append(0)

        return str(recall)

    def get_test_precision_per_class(self):
        precision = []
        for i in range(len(self.classes)):
            if (self.TP[i] + self.FP[i]) != 0:
                precision.append(self.TP[i] / (self.TP[i] + self.FP[i]))
            else:
                precision.append(0)
        return str(precision)

    def calculate_fpr_tpr_th(self):
        predicted_labels_proba = self.classification.predict_proba(self.test_data)
        for each in self.classes:
            class_num = self.classes.index(each)
            fpr, tpr, th = roc_curve(self.test_labels, predicted_labels_proba[:, class_num], pos_label=class_num)
            self.fpr.append(fpr)
            self.tpr.append(tpr)
            self.th.append(th)

        # self.fpr = np.array(self.fpr)
        # self.tpr = np.array(self.tpr)
        # self.th = np.array(self.th)

    def plot_ROC_curve(self):
        self.calculate_fpr_tpr_th()
        font = {'size': 16, 'weight': 'normal'}
        plt.close()
        plt.figure()

        plt.xlabel("False Positive Rate", fontdict=font)
        plt.ylabel("True Positive Rate", fontdict=font)
        plt.title("ROC Curve", fontdict=font)
        plt.xlim([0, 1])
        plt.ylim([0, 1.05])

        for each in self.classes:
            class_num = self.classes.index(each)
            plt.plot(self.fpr[class_num], self.tpr[class_num], label=each)

        plt.legend(loc='lower right')
        plt.tight_layout()
        # plt.show()
        plt.savefig("D:/Licenta/Interfata/resources/images/ROC_curve_RF.png", dpi=500)

    def get_classes(self):
        return str(self.classes)

    def print_path(self):
        return self.data_path

    def class_rep(self):
        print(classification_report(self.test_labels, self.predicted_labels))


# clf = ClassificationRandomForest()
# clf.set_data_path("D:/Licenta/complete_dataset")
# clf.set_images_and_labels()
# clf.encode_labels()
# # print(str(clf.get_classes()))
# clf.normalize_data()
# clf.split_data()
# clf.train()
# print("train acc:", clf.get_train_accuracy())
# print("test acc:", clf.get_test_accuracy())
# clf.generate_confusion_matrix()
# print(clf.confusion_matrix)
# clf.normalize_confusion_matrix()
# clf.plot_confusion_matrix()
# clf.get_values(clf.confusion_matrix, clf.classes)
# print(clf.get_test_accuracy_oonf_matr())
# print(clf.get_test_recall_per_class())
# print(clf.get_test_precision_per_class())
#
# # print(clf.tpr[0][:3],clf.fpr,clf.th)
# clf.plot_ROC_curve()
# clf.class_rep()