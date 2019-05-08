import os
import cv2
import csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, classification_report, precision_recall_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, normalize
from sklearn.model_selection import train_test_split
from sklearn import utils
from skimage.feature import greycomatrix, greycoprops
import pandas as pd
import matplotlib
from sklearn.model_selection import GridSearchCV
import time


class ClassificationRandomForest:
    def __init__(self):
        self.dir_list = None
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

        self.precision = []
        self.recall = []

        self.fpr = []
        self.tpr = []
        self.th = []


    def save_data(self, data):
        f = open("D:/Licenta/Interfata/resources/results/random_forest_results.csv", 'a', newline='')
        csv.writer(f).writerow(data)
        f.close()

    def set_data_path(self, path):
        self.data_path = path

    def extract_features(self, image):
        (B, G, R) = cv2.split(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        (H, S, V) = cv2.split(image)
        return [np.mean(B), np.mean(G), np.mean(R), np.mean(H), np.mean(S), np.std(B), np.std(G), np.std(R),
                    np.std(H), np.std(S)]

    def set_images_and_labels(self):
        #img_list = []
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
        start = time.time()
        self.classification.fit(self.train_data, self.train_labels)
        pickle.dump(self.classification, open('D:/Licenta/Models/rf.sav', 'wb'))
        end = time.time()
        self.save_data([end - start])
        print("Execution time:", end - start, "seconds")

        # print(grid.best_params_)
        # feature_imp = pd.Series(cl.feature_importances_,index=["b mean", "g mean", "r mean", "b dev", "g dev",
        # "r dev"]).sort_values(ascending=False)

    def load(self):
        self.classification =  pickle.load(open('D:/Licenta/Models/rf.sav', 'rb'))

    def get_train_accuracy(self):
        # Calcul acuratete pentru datele de antrenare
        return accuracy_score(self.train_labels, self.classification.predict(self.train_data))

    def predict(self):
        # Testing
        self.predicted_labels = self.classification.predict(self.test_data)

    def get_test_accuracy(self):
        # self.cm_classes = np.union1d(self.test_labels, self.predicted_labels)
        self.save_data([accuracy_score(self.test_labels, self.predicted_labels)])
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
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        # Creare figura si axe
        fig, ax = plt.subplots()

        # Heatmap
        im = ax.imshow(self.normalized_confusion_matrix, cmap='summer')

        # Colorbar
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
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

        legend = []
        for each in range(len(self.classes)):
            legend.append(' '.join([str(each), '-', self.classes[each]]))
        textstr = '\n'.join(legend)

        ax.text(7.5, 2.5, textstr, fontsize=11, va='center', linespacing=1.5)
        #bbox=dict(boxstyle='square', pad=1, fill=False)


        # Ajustare automata a parametrilor figurii
        #fig.tight_layout()
        fig.savefig("D:/Licenta/Interfata/resources/results/confusion_matrix_RF.png", bbox_inches='tight', orientation='landscape', transparent=True, dpi=500)
        # plt.show()


    def get_values(self, matrix, classes):
        self.total_test_samples = np.sum(matrix[:, :])

        self.TP = []
        self.TN = []
        self.FN = []
        self.FP = []

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



    def get_test_recall_per_class(self):
        self.get_values(self.confusion_matrix, self.classes)
        recall = []
        for i in range(len(self.classes)):
            if (self.TP[i] + self.FN[i]) != 0:
                recall.append(self.TP[i] / (self.TP[i] + self.FN[i]))
            else:
                recall.append(0)
        self.save_data(recall)
        return str(recall)

    def get_test_precision_per_class(self):
        precision = []
        for i in range(len(self.classes)):
            if (self.TP[i] + self.FP[i]) != 0:
                precision.append(self.TP[i] / (self.TP[i] + self.FP[i]))
            else:
                precision.append(0)
        self.save_data(precision)
        return str(precision)

    def get_test_accuracy_conf_matr(self):
        acc = np.sum(self.TP) / self.total_test_samples
        self.save_data([acc])
        return acc

    def calculate_fpr_tpr_th(self):
        predicted_labels_proba = self.classification.predict_proba(self.test_data)
        for each in self.classes:
            class_num = self.classes.index(each)
            fpr, tpr, th = roc_curve(self.test_labels, predicted_labels_proba[:, class_num], pos_label=class_num)
            prec, rec, th2 = precision_recall_curve(self.test_labels, predicted_labels_proba[:, class_num], pos_label=class_num)

            self.fpr.append(fpr)
            self.tpr.append(tpr)
            self.th.append(th)
            self.precision.append(prec)
            self.recall.append(rec)

        # self.fpr = np.array(self.fpr)
        # self.tpr = np.array(self.tpr)
        # self.th = np.array(self.th)

    def plot_ROC_curve(self):
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
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
        plt.savefig("D:/Licenta/Interfata/resources/results/ROC_curve_RF.png", dpi=500)

    def plot_PR_curve(self):
        from matplotlib import pyplot as plt
        font = {'size': 16, 'weight': 'normal'}
        plt.close()
        plt.figure()

        plt.xlabel("Recall", fontdict=font)
        plt.ylabel("Precision", fontdict=font)
        plt.title("Precision Recall Curve", fontdict=font)
        plt.xlim([0, 1])
        plt.ylim([0, 1.05])

        for each in self.classes:
            class_num = self.classes.index(each)
            plt.plot(self.recall[class_num], self.precision[class_num], label=each)

        plt.legend(loc='lower right')
        plt.tight_layout()
        # plt.show()
        plt.savefig("D:/Licenta/Interfata/resources/results/PR_curve_SVM.png", dpi=500)

    def get_classes(self):
        return str(self.classes)

    def print_path(self):
        return self.data_path

    def class_rep(self):
        print(classification_report(self.test_labels, self.predicted_labels))


# clf = ClassificationRandomForest()
#
# clf.set_data_path("../images/dataset")
# #clf.set_data_path("D:/train")
# clf.set_images_and_labels()
# clf.encode_labels()
# clf.normalize_data()
# clf.split_data()
# clf.train()
# clf.predict()
# print("train acc:", clf.get_train_accuracy())
# print("test acc:", clf.get_test_accuracy())
# clf.generate_confusion_matrix()
# print(clf.confusion_matrix)
# clf.normalize_confusion_matrix()
# clf.plot_confusion_matrix()
# clf.get_values(clf.confusion_matrix, clf.classes)
# print(clf.get_test_accuracy_conf_matr())
# print(clf.get_test_recall_per_class())
# print(clf.get_test_precision_per_class())
# # print(str(clf.get_classes()))
# # print(clf.tpr[0][:3],clf.fpr,clf.th)
# clf.plot_ROC_curve()
# clf.plot_PR_curve()
# clf.class_rep()