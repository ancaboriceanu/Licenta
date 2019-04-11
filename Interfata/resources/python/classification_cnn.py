import os
import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras import optimizers
from keras import applications
from keras.models import Model
from sklearn.metrics import confusion_matrix, roc_curve
import cv2
import matplotlib.pyplot as plt
import glob


class ClassificationCNN:
    def __init__(self):
        self.train_data_path = ''
        self.validation_data_path = ''
        self.test_data_path = ''

        self.classes = []
        self.test_labels = []

        self.img_width, self.img_height = 128, 128

        self.train_samples = 9632
        self.validation_samples = 1184
        self.test_samples = 1184

        # self.train_samples = 41
        # self.validation_samples = 41
        # self.test_samples = 41

        self.predicted_labels = []

        self.batch_size = 32
        self.epochs = 1

        self.model = None

        self.confusion_matrix = np.zeros((6, 6), dtype=int)
        self.normalized_confusion_matrix = np.zeros((6, 6), dtype=float)

        self.history = None

        self.datagen = ImageDataGenerator(rescale=1./255)

        self.TP = []
        self.TN = []
        self.FN = []
        self.FP = []

        self.fpr = []
        self.tpr = []
        self.th = []

    def set_data_path(self, train_data_path, validation_data_path, test_data_path):
        self.train_data_path = train_data_path
        self.validation_data_path = validation_data_path
        self.test_data_path = test_data_path

    def create_model(self):
        self.model = Sequential()

        self.model.add(Convolution2D(64, (3, 3), input_shape=(self.img_width, self.img_height, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Convolution2D(32, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.5))

        self.model.add(Convolution2D(16, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(32))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(6))
        self.model.add(Activation('softmax'))

        self.model.compile(loss='binary_crossentropy',
                           optimizer='rmsprop',
                           metrics=['accuracy'])

    def train(self):
        from PIL import ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        train_generator = self.datagen.flow_from_directory(
            self.train_data_path,
            target_size=(self.img_width, self.img_height),
            batch_size=self.batch_size,
            class_mode='categorical')

        classes_dict = train_generator.class_indices
        self.classes = list(classes_dict.keys())
        self.classes.sort()

        validation_generator = self.datagen.flow_from_directory(
            self.validation_data_path,
            target_size=(self.img_width, self.img_height),
            batch_size=self.batch_size,
            class_mode='categorical')



        self.history = self.model.fit_generator(train_generator, epochs=self.epochs,
                                                steps_per_epoch=self.train_samples // self.batch_size,
                                                validation_data=validation_generator,
                                                validation_steps=self.validation_samples // self.batch_size)

        self.model.save('D:/Licenta/Models/cnn_model.h5')

    def get_loss_and_accuracy(self):
        test_generator = self.datagen.flow_from_directory(
            self.test_data_path,
            target_size=(self.img_width, self.img_height),
            batch_size=self.batch_size, shuffle=False,
            class_mode='categorical')

        self.test_data, self.test_labels_categorical = next(test_generator)
        self.test_labels = np.argmax(self.test_labels_categorical, axis = 1)
        print(self.test_labels)
        #print("test labels ordonate", self.test_labels)

        self.predicted_labels_proba = self.model.predict_generator(test_generator, steps=1, workers=0)

        self.predicted_labels = np.argmax(self.predicted_labels_proba, axis=1)
        metr = self.model.evaluate_generator(test_generator, steps=1)
        print("predicted", self.predicted_labels)
        print("predicted proba", self.predicted_labels_proba)
        #print(self.predicted_labels)

        return [metr[0], metr[1]]

    def plot_loss(self):
        history_dict = self.history.history
        loss_values = history_dict['loss']
        val_loss_values = history_dict['val_loss']

        epochs = range(self.epochs)

        plt.plot(epochs, loss_values, 'b-', label='Training loss')
        plt.plot(epochs, val_loss_values, 'm-', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.savefig("D:/Licenta/Interfata/resources/images/loss_CNN.png")

    def plot_accuracy(self):
        plt.clf()
        history_dict = self.history.history
        acc_values = history_dict['acc']
        val_acc_values = history_dict['val_acc']

        epochs = range(self.epochs)

        plt.plot(epochs, acc_values, 'g-', label='Training accuracy')
        plt.plot(epochs, val_acc_values, 'r-', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.savefig("D:/Licenta/Interfata/resources/images/accuracy_CNN.png", dpi=500)

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

        ax.set_xticklabels(self.classes,  rotation=45, ha="right", rotation_mode="anchor")
        ax.set_yticklabels(self.classes)

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

        # Ajustare automata a parametrilor figurii
        fig.tight_layout()
        fig.savefig("D:/Licenta/Interfata/resources/images/confusion_matrix_CNN.png", dpi=500)
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
        recall = []
        for i in range(len(self.classes)):
            if (self.TP[i] + self.FN[i]) != 0:
                recall.append(self.TP[i] / (self.TP[i] + self.FN[i]))
            else:
                recall.append(0)
        return recall

    def get_test_precision_per_class(self):
        precision = []
        for i in range(len(self.classes)):
            if (self.TP[i] + self.FP[i]) != 0:
                precision.append(self.TP[i] / (self.TP[i] + self.FP[i]))
            else:
                precision.append(0)
        return precision

    def calculate_fpr_tpr_th(self):
        for each in self.classes:
            class_num = self.classes.index(each)
            fpr, tpr, th = roc_curve(self.test_labels, self.predicted_labels_proba[:, class_num], pos_label=class_num)
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
        plt.savefig("D:/Licenta/Interfata/resources/images/ROC_curve_CNN.png", dpi=500)


# train_data_dir = 'D:/train'
# validation_data_dir = 'D:/train'
# test_data_dir = 'D:/train'

train_data_dir = 'D:/Licenta/newdataset/train'
validation_data_dir = 'D:/Licenta/newdataset/validation'
test_data_dir = 'D:/Licenta/newdataset/test'

clf = ClassificationCNN()
clf.set_data_path(train_data_dir, validation_data_dir, test_data_dir)
clf.create_model()
clf.train()
print(clf.get_loss_and_accuracy())
clf.plot_loss()
clf.plot_accuracy()
clf.generate_confusion_matrix()
clf.normalize_confusion_matrix()
clf.plot_confusion_matrix()
clf.get_values(clf.confusion_matrix, clf.classes)
print(clf.confusion_matrix)
print(clf.get_test_accuracy_oonf_matr())
print(clf.get_test_recall_per_class())
print(clf.get_test_precision_per_class())

clf.plot_ROC_curve()
