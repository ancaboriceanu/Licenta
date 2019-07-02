#ifndef SUPPORTVECTORMACHINESCLASSIFICATION_H
#define SUPPORTVECTORMACHINESCLASSIFICATION_H

#include <QObject>
#include <PythonQt.h>
#include <QThread>
#include <QList>
#include <Classification.h>

class SupportVectorMachinesClassification : public QObject, public Classification
{
    Q_OBJECT
    Q_INTERFACES(Classification)

public:
    explicit SupportVectorMachinesClassification(QObject *parent = nullptr);
public:
    void setPath(QString dataPath);

signals:
    void donePreprocessing();
    void trainFinished();
    void results(float trainAcc, float testAcc, QList<float> recall, QList<float> precision, QStringList classes);

public slots:
    void process();
    void train();
    void test(){}


private:
    QString m_dataPath;
    PythonQtObjectPtr m_mainContext;
    PythonQtObjectPtr m_tag;
};

#endif // SUPPORTVECTORMACHINESCLASSIFICATION_H
