#ifndef SUPPORTVECTORMACHINESCLASSIFICATION_H
#define SUPPORTVECTORMACHINESCLASSIFICATION_H

#include <QObject>
#include <PythonQt.h>
#include <QThread>
#include <QList>

class SupportVectorMachinesClassification : public QObject
{
    Q_OBJECT
public:
    explicit SupportVectorMachinesClassification(QObject *parent = 0);

public:
    void setPath(QString dataPath);

signals:
    void donePreprocessing();
    void finished();
    void svmResults(float trainAcc, float testAcc, QList<float> recall, QList<float> precision, QStringList classes);

public slots:
    void process();
    void train();

private:
    QList<float> convertString(QString string);
    QStringList splitString(QString string);

private:
    QString m_dataPath;
    PythonQtObjectPtr m_mainContext;
    PythonQtObjectPtr m_tag;
};

#endif // SUPPORTVECTORMACHINESCLASSIFICATION_H
