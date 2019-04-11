#ifndef CONVOLUTIONALNEURALNETWORKSCLASSIFICATION_H
#define CONVOLUTIONALNEURALNETWORKSCLASSIFICATION_H

#include <QObject>
#include <QDebug>
#include <QThread>
#include <PythonQt.h>

class ConvolutionalNeuralNetworksClassification : public QObject
{
    Q_OBJECT
public:
    explicit ConvolutionalNeuralNetworksClassification(QObject *parent = 0);

public:
    void setPath(QString dataPath);

signals:
    void donePreprocessing();
    void finished();

public slots:
    void process();
    void train();

private:
    QString m_dataPath;
    PythonQtObjectPtr m_mainContext;
    PythonQtObjectPtr m_tag;
};

#endif // CONVOLUTIONALNEURALNETWORKSCLASSIFICATION_H
