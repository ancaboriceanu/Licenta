#ifndef SVM_H
#define SVM_H

#include "supportvectormachinesclassification.h"
#include <QThread>
#include <QWidget>

#include <QtWidgets/QMainWindow>
#include <QtCharts/QChartView>
#include <QtCharts/QBarSet>
#include <QtCharts/QBarSeries>
#include <QtCharts/QLegend>
#include <QtCharts/QBarCategoryAxis>
#include <QtCharts/QLineSeries>
#include <QtCharts/QHorizontalStackedBarSeries>
#include <QtCharts/QCategoryAxis>

#include <QtCharts/QPieSeries>
#include <QtCharts/QPieSlice>

QT_CHARTS_USE_NAMESPACE

namespace Ui {


class svm;
}

class svm : public QWidget
{
    Q_OBJECT


public:
    explicit svm(QWidget *parent = 0, QString dataPath = "");
    ~svm();

signals:
    void backClicked();
    void compareClicked(QStringList classes);
    void startTrainingClicked();

private slots:
    void initialization();
    void updatePreprocessingFinished();
    void updateStartTraining();
    void updateFinished();
    void on_startTrainingButton_clicked();
    void on_backButton_clicked();
    void on_compareButton_clicked();
    void on_confusionMatrixButton_clicked();
    void on_RocCurveButton_clicked();
    void setResults(float trainAcc, float testAcc, QList<float> recall, QList<float> precision, QStringList classes);
    void on_accuracyButton_clicked();
    void on_precRecallButton_clicked();
    void on_showResultsButton_clicked();
    void showValue(bool status,int index, QBarSet *barSet);


private:
    void setWindow();
    void plotPrecisionRecall();
    void plotAccuracy();

private:
    Ui::svm *ui;

    float m_trainAcc;
    float m_testAcc;
    QList<float> m_recall;
    QList<float> m_precision;
    QStringList m_classes;

    QThread* classificationThread;



};

#endif // SVM_H
