#ifndef CONVOLUTIONALNEURALNETWORKSCLASSIFICATION_H
#define CONVOLUTIONALNEURALNETWORKSCLASSIFICATION_H

#include <QObject>
#include <QDebug>
#include <QThread>
#include <PythonQt.h>
#include <QDir>

class ConvolutionalNeuralNetworksClassification : public QObject
{
    Q_OBJECT
public:
    explicit ConvolutionalNeuralNetworksClassification(QObject *parent = 0);

public:
    void setPath(QString dataPath);

signals:
    void trainFinished();
    void testFinished();
    void cnnResults(float testAcc, QList<float> recall, QList<float> precision, QStringList classes);

public slots:
    void train();
    void test();


private:
    QList<float> convertString(QString string);
    QStringList splitString(QString string);

private:
    QString m_dataPath;
};

#endif // CONVOLUTIONALNEURALNETWORKSCLASSIFICATION_H
