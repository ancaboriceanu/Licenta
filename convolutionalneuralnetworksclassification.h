#ifndef CONVOLUTIONALNEURALNETWORKSCLASSIFICATION_H
#define CONVOLUTIONALNEURALNETWORKSCLASSIFICATION_H

#include <QObject>
#include <QDebug>
#include <QThread>
#include <PythonQt.h>
#include <QDir>
#include <Classification.h>

class ConvolutionalNeuralNetworksClassification : public QObject, public Classification
{
    Q_OBJECT
    Q_INTERFACES(Classification)

public:
    explicit ConvolutionalNeuralNetworksClassification(QObject *parent = nullptr);

public:
    void setPath(QString dataPath);

signals:
    void trainFinished();
    void testFinished();
    void results(float testAcc, QList<float> recall, QList<float> precision, QStringList classes);

public slots:
    void process(){}
    void train();
    void test();

private:
    QString m_dataPath;
    PythonQtObjectPtr m_mainContext;
    PythonQtObjectPtr m_tag;
};

#endif // CONVOLUTIONALNEURALNETWORKSCLASSIFICATION_H
