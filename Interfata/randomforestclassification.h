#ifndef RANDOMFORESTCLASSIFICATION_H
#define RANDOMFORESTCLASSIFICATION_H

#include <QObject>
#include <QDebug>
#include <QThread>
#include <PythonQt.h>


class RandomForestClassification : public QObject
{
    Q_OBJECT
public:
    explicit RandomForestClassification(QObject *parent = 0);

public:
    void setPath(QString dataPath);

signals:
    void donePreprocessing();
    void finished();
    void randomForestResults(float trainAcc, float testAcc, QList<float> recall, QList<float> precision, QStringList classes);

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

#endif // RANDOMFORESTCLASSIFICATION_H
