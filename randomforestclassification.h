#ifndef RANDOMFORESTCLASSIFICATION_H
#define RANDOMFORESTCLASSIFICATION_H

#include <QObject>
#include <QDebug>
#include <QThread>
#include <PythonQt.h>
#include <Classification.h>

class RandomForestClassification : public QObject, public Classification
{
    Q_OBJECT
    Q_INTERFACES(Classification)

public:
    explicit RandomForestClassification(QObject *parent = nullptr);

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

#endif // RANDOMFORESTCLASSIFICATION_H
