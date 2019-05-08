#-------------------------------------------------
#
# Project created by QtCreator 2019-03-29T22:01:45
#
#-------------------------------------------------

QT       += core gui charts

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets charts printsupport

TARGET = Interfata
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp \
    start.cpp \
    cnn.cpp \
    svm.cpp \
    final.cpp \
    randomforestclassification.cpp \
    supportvectormachinesclassification.cpp \
    random_forest.cpp \
    convolutionalneuralnetworksclassification.cpp \
    dynamicplot.cpp

HEADERS  += mainwindow.h \
    start.h \
    cnn.h \
    svm.h \
    final.h \
    randomforestclassification.h \
    supportvectormachinesclassification.h \
    random_forest.h \
    convolutionalneuralnetworksclassification.h \
    dynamicplot.h

FORMS    += mainwindow.ui \
    start.ui \
    cnn.ui \
    svm.ui \
    final.ui \
    random_forest.ui

RESOURCES += resources/python/classification_svm.py \
             resources/python/classification_random_forest.py \
             resources/python/classification_cnn.py \
             resources/results


win32:CONFIG(release, debug|release): LIBS += -LC:/Users/ancab/AppData/Local/Programs/Python/Python35/libs/ -lpython35
else:win32:CONFIG(debug, debug|release): LIBS += -LC:/Users/ancab/AppData/Local/Programs/Python/Python35/libs/ -lpython35d
else:unix: LIBS += -LC:/Users/ancab/AppData/Local/Programs/Python/Python35/libs/ -lpython35

INCLUDEPATH += C:/Users/ancab/AppData/Local/Programs/Python/Python35/include
DEPENDPATH += C:/Users/ancab/AppData/Local/Programs/Python/Python35/include

win32:CONFIG(release, debug|release): LIBS += -LD:/PythonQt3.2/lib/ -lPythonQt-Qt5-Python3
else:win32:CONFIG(debug, debug|release): LIBS += -LD:/PythonQt3.2/lib/ -lPythonQt-Qt5-Python3d
else:unix: LIBS += -LD:/PythonQt3.2/lib/ -lPythonQt-Qt5-Python3

INCLUDEPATH += D:/PythonQt3.2/src
DEPENDPATH += D:/PythonQt3.2/src

DISTFILES +=
