import os
import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense,BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras import optimizers
from keras import applications
from keras.models import Model
from sklearn.metrics import confusion_matrix, roc_curve, classification_report
import cv2
import matplotlib
import glob
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
from keras.metrics import categorical_accuracy, sparse_categorical_accuracy
from keras.callbacks import Callback
import os
import csv
import time

class GetMetrics(Callback):
    def plot(self):
        import matplotlib.pyplot as plt

        #plt.ion()
        #plt.show()

        plt.plot(self.epochs, self.losses, 'b-', label='Training loss')
        plt.plot(self.epochs, self.accuracy, 'm-', label='Training accuracy')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        #plt.pause(0.001)
        plt.savefig("D:/Licenta/Interfata/resources/results/CNN.png")

    def on_train_begin(self, logs=None):
        self.losses = []
        self.accuracy = []
        self.epochs = []

        self.epoch = 0
        self.nr_batches = 301

        #f = open("D:/Licenta/Interfata/resources/results/cnn_plot_values.csv", 'w', newline='')

    def write(self, file, row):
        f = open(file, 'a', newline='')
        csv.writer(f).writerow(row)
        f.close()

    def on_batch_end(self, batch, logs=None):
        values = [(self.epoch * self.nr_batches) + batch + 1, logs.get('loss'), logs.get('acc'), self.epoch + 1]
        self.write("D:/Licenta/Interfata/resources/results/cnn_plot_values.csv", values)

    def on_epoch_end(self, epoch, logs=None):
        self.epoch = self.epoch + 1


    """ 
    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get('loss'))
        self.accuracy.append(logs.get('acc'))
        self.epochs.append(epoch)
        values = [epoch + 1, logs.get('loss'), logs.get('acc')]
        self.write("D:/Licenta/Interfata/resources/results/cnn_plot_values.csv", values)
        self.plot()
        #print("End of epoch:", epoch + 1)
        #print("Metrics:", logs)"""


class ClassificationCNN():
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

        self.predicted_labels = []

        self.batch_size = 32
        self.epochs = 100

        self.model = None

        self.confusion_matrix = np.zeros((6, 6), dtype=int)
        self.normalized_confusion_matrix = np.zeros((6, 6), dtype=float)

        self.History = None

        self.datagen = ImageDataGenerator(rescale=1./255)



        self.fpr = []
        self.tpr = []
        self.th = []

        self.TP = []
        self.TN = []
        self.FN = []
        self.FP = []

        self.total_test_samples = 0


    def save_data(self, file, data):
        f = open(file, 'a', newline='')
        csv.writer(f).writerow(data)
        f.close()

    def set_data_path(self, data_path):
        self.train_data_path = os.path.join(data_path, "train")
        self.validation_data_path = os.path.join(data_path, "validation")
        self.test_data_path = os.path.join(data_path, "test")


    def set_generators(self):

        self.train_generator = self.datagen.flow_from_directory(
            self.train_data_path,
            target_size=(self.img_width, self.img_height),
            batch_size=self.batch_size,
            class_mode='categorical')

        self.validation_generator = self.datagen.flow_from_directory(
            self.validation_data_path,
            target_size=(self.img_width, self.img_height),
            batch_size=self.batch_size,
            class_mode='categorical')

        self.test_generator = self.datagen.flow_from_directory(
            self.test_data_path,
            target_size=(self.img_width, self.img_height),
            batch_size=self.batch_size, shuffle=False,
            class_mode='categorical')

        classes_dict = self.train_generator.class_indices
        self.classes = list(classes_dict.keys())
        self.classes.\
            sort()

    def create_model(self):
        self.model = Sequential()

        self.model.add(Convolution2D(128, (3, 3), padding='same', input_shape=(self.img_width, self.img_height, 3)))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(128, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.3))

        self.model.add(Convolution2D(64, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(64, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Convolution2D(32, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(32, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.5))

        self.model.add(Convolution2D(16, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(16, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(512))  # mult mai mult
        self.model.add(Activation('relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.5))
        self.model.add(Dense(6))
        self.model.add(Activation('softmax'))

        print(self.model.summary())

        self.model.compile(loss='binary_crossentropy',
                           optimizer='rmsprop',
                           metrics=['accuracy'])


    def train(self):
        from PIL import ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        gm = GetMetrics()

        #Early Stopping
        es = EarlyStopping(monitor='val_acc', mode='max', patience=30, verbose=1, min_delta=0.5)
        mc = ModelCheckpoint('D:/Licenta/Models/best_cnn.h5', monitor='val_acc', mode='max',
                             save_best_only=True, save_weights_only=True, verbose=1)

        start = time.time()
        self.History = self.model.fit_generator(self.train_generator, epochs=self.epochs,
                                                steps_per_epoch=self.train_samples // self.batch_size,
                                                validation_data=self.validation_generator,
                                                validation_steps=self.validation_samples // self.batch_size,
                                                callbacks=[gm, es, mc], verbose=1)

        end = time.time()
        print("Execution time:", end - start, "seconds")
        #self.model.save('D:/Licenta/Models/cnn_model.h5')

        print("training stopped at epoch:", es.stopped_epoch)
        acc = np.mean(self.History.history['acc'])
        # if(os.path.isfile("D:/Licenta/Interfata/resources/results/cnn_train_acc.csv")):
        #     os.remove("D:/Licenta/Interfata/resources/results/cnn_train_acc.csv")
        self.save_data("D:/Licenta/Interfata/resources/results/cnn_train_acc.csv", [acc])
        self.save_data("D:/Licenta/Interfata/resources/results/cnn_train_acc.csv", [end - start])


    def get_classes(self):
        return str(self.classes)

    def load(self):
        #del self.model
        self.model.load_weights('D:/Licenta/Models/best_cnn.h5')

    def get_test_accuracy(self):

        self.test_labels = self.test_generator.classes


        print("test labels:", self.test_labels)

        self.predicted_labels_proba = self.model.predict_generator(self.test_generator,
                                                                   steps=self.test_samples // self.batch_size)

        self.predicted_labels = np.argmax(self.predicted_labels_proba, axis=-1)

        metr = self.model.evaluate_generator(self.test_generator,
                                             steps=self.test_samples // self.batch_size)

        print(metr)

        print("predicted labels:", self.predicted_labels)

        print("Sklearn acc:", accuracy_score(self.test_labels, self.predicted_labels))

        return metr[1]

    def plot_loss(self):
        import matplotlib.pyplot as plt
        history_dict = self.History.history
        loss_values = history_dict['loss']
        val_loss_values = history_dict['val_loss']

        epochs = range(len(self.History.history['loss']))

        plt.plot(epochs, loss_values, 'b-', label='Training loss')
        plt.plot(epochs, val_loss_values, 'm-', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.savefig("D:/Licenta/Interfata/resources/results/loss_CNN.png")

    def plot_accuracy(self):
        import matplotlib.pyplot as plt

        plt.clf()
        history_dict = self.History.history
        acc_values = history_dict['acc']
        val_acc_values = history_dict['val_acc']

        epochs = range(len(self.History.history['acc']))

        plt.plot(epochs, acc_values, 'g-', label='Training accuracy')
        plt.plot(epochs, val_acc_values, 'r-', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.savefig("D:/Licenta/Interfata/resources/results/accuracy_CNN.png", dpi=500)

    def generate_confusion_matrix(self):
        self.confusion_matrix = confusion_matrix(self.test_labels, self.predicted_labels)
        return str((self.test_labels, self.predicted_labels))

    def normalize_confusion_matrix(self):
        suma = self.confusion_matrix.sum(axis=1)
        for i in range(len(self.classes)):
            for j in range(len(self.classes)):
                if suma[i] > 0:
                    self.normalized_confusion_matrix[i, j] = self.confusion_matrix.astype('float')[i, j] / suma[i]

    def plot_confusion_matrix(self):
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

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

        legend = []
        for each in range(len(self.classes)):
            legend.append(' '.join([str(each), '-', self.classes[each]]))
        textstr = '\n'.join(legend)

        ax.text(7.5, 2.5, textstr, fontsize=11, va='center', linespacing=1.5)
        # bbox=dict(boxstyle='square', pad=1, fill=False)

        # Ajustare automata a parametrilor figurii
        # fig.tight_layout()
        fig.savefig("D:/Licenta/Interfata/resources/results/confusion_matrix_CNN.png", bbox_inches='tight',
                    orientation='landscape', transparent=True)
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



    def get_test_recall_per_class(self):
        self.get_values(self.confusion_matrix, self.classes)

        recall = []
        for i in range(len(self.classes)):
            if (self.TP[i] + self.FN[i]) != 0:
                recall.append(self.TP[i] / (self.TP[i] + self.FN[i]))
            else:
                recall.append(0)
        self.save_data("D:/Licenta/Interfata/resources/results/cnn_results.csv", recall)
        print("hello")

        return str(recall)

    def get_test_precision_per_class(self):
        precision = []
        for i in range(len(self.classes)):
            if (self.TP[i] + self.FP[i]) != 0:
                precision.append(self.TP[i] / (self.TP[i] + self.FP[i]))
            else:
                precision.append(0)
        self.save_data("D:/Licenta/Interfata/resources/results/cnn_results.csv", precision)
        return str(precision)

    def get_test_accuracy_conf_matr(self):
        acc = np.sum(self.TP) / self.total_test_samples
        self.save_data("D:/Licenta/Interfata/resources/results/cnn_results.csv", [acc])
        return acc

    def evaluate_th(self):
        for c in range(len(self.th)):
            for each in range(len(self.th[c])):
                ac = 1 - self.tpr[c][each]
                print("Acc for threshold",self.th[c][each],"is",ac)

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
        #self.evaluate_th()
    def plot_ROC_curve(self):
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
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
        plt.savefig("D:/Licenta/Interfata/resources/results/ROC_curve_CNN.png", dpi=500)

    def class_rep(self):
        print(classification_report(self.test_labels, self.predicted_labels))


# clf = ClassificationCNN()
# clf.set_data_path('../images/cnn_dataset')
# clf.set_generators()
# clf.create_model()
# #clf.train()
# clf.load()
# print(clf.get_test_accuracy())
# # clf.plot_loss()
# # clf.plot_accuracy()
# clf.generate_confusion_matrix()
# clf.normalize_confusion_matrix()
# clf.plot_confusion_matrix()
# print(clf.confusion_matrix)
#
# print(clf.get_test_recall_per_class())
# print(clf.get_test_precision_per_class())
# print(clf.get_test_accuracy_conf_matr())
#
#
#
# print("tp:", clf.TP)
# print("tn:", clf.TN)
# print("fp:", clf.FP)
# print("fn:", clf.FN)
# print("TOTAL:", clf.total_test_samples)
# clf.class_rep()
# clf.plot_ROC_curve()
