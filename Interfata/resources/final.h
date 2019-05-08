#ifndef FINAL_H
#define FINAL_H
#include "mainwindow.h"
#include <QMetaType>
#include <QVector>

#include "cnn.h"
#include "random_forest.h"
#include "svm.h"

#include <QWidget>

//struct ResultsStruct{
//public:
//    ResultsStruct(){}
//    ResultsStruct(QList<float> recall, QList<float> precision, float testAcc)
//    {
//        this->recall = recall;
//        this->precision = precision;
//        this->testAcc = testAcc;

//    }
//    ~ResultsStruct(){}

//    float testAcc;
//    QList<float> recall;
//    QList<float> precision;
//};

//Q_DECLARE_METATYPE(ResultsStruct)

namespace Ui {
class final;
}

class final : public QWidget
{
    Q_OBJECT

public:
    explicit final(QWidget *parent = 0, QStringList classes = {});
    ~final();

signals:
    void homeClicked();

private slots:
    void on_nextToolButton_clicked();

    void on_backToolButton_clicked();

    void on_homeToolButton_clicked();

private:
    void plotAccuracy();
    void createTable();
    void plotTime();
    QList<float> splitString(QString text);
    QList<QList<float> > readFromFile(QString file);

private:
    Ui::final *ui;

private:
    QList<QList<float> > m_cnnResults;
    QList<QList<float> > m_svmResults;
    QList<QList<float> > m_rfResults;
    QStringList m_classes;

};

#endif // FINAL_H
