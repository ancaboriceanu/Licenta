#ifndef CLASSIFICATION_H
#define CLASSIFICATION_H
#include <QtCore>

#include <PythonQt.h>
#include <QDebug>


class Classification
{
public:
    Classification();

public:
    virtual ~Classification(){}
    virtual void setPath(QString dataPath) = 0;


public slots:
    virtual void process() = 0;
    virtual void train() = 0;
    virtual void test() = 0;

private:
    QString m_dataPath;
    PythonQtObjectPtr m_mainContext;
    PythonQtObjectPtr m_tag;
};

Q_DECLARE_INTERFACE(Classification, "Classification")

#endif // CLASSIFICATION_H
