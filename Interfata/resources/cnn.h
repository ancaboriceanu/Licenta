#ifndef CNN_H
#define CNN_H

#include "convolutionalneuralnetworksclassification.h"
#include "final.h"
#include "dynamicplot.h"


#include <QWidget>
#include <QTimer>
#include <QChartView>

#include <QtWidgets/QApplication>
#include <QtWidgets/QMainWindow>
#include <QtCharts/QChartView>
#include <QtCharts/QBarSet>
#include <QtCharts/QBarSeries>
#include <QtCharts/QLegend>
#include <QtCharts/QBarCategoryAxis>
#include <QtCharts/QLineSeries>
#include <QtCharts/QHorizontalStackedBarSeries>
#include <QtCharts/QCategoryAxis>


namespace Ui {
class cnn;
}

class cnn : public QWidget
{
    Q_OBJECT


public:
    explicit cnn(QWidget *parent = 0, QString dataPath = "");
    ~cnn();

signals:
    void backClicked();
    void compareClicked(QStringList classes);
    void startTrainingClicked();
    void startTestingClicked();

private slots:
    void initialization();
    void updateStartTraining();
    void updateStartTesting();
    void updateTrainFinished();
    void updateTestFinished();
    void on_backButton_clicked();
    void on_compareButton_clicked();

    void on_confusionMatrixButton_clicked();

    void on_RocCurveButton_clicked();

    void on_showResultsButton_clicked();
    void updatePlot();
    void setResults(float testAcc, QList<float> recall, QList<float> precision, QStringList classes);
    void on_accuracyButton_clicked();

    void on_precRecallButton_clicked();

    void showValue(bool status,int index, QBarSet *barSet);

    void on_trainRadioButton_clicked();

    void on_testRadioButton_clicked();

private:
    void setWindow();
    QList<float> splitString(QString text);
    void plotPrecisionRecall();
    void plotAccuracy();

private:
    Ui::cnn *ui;
    QTimer *m_timer;

    float m_trainAcc;
    float m_testAcc;
    QList<float> m_recall;
    QList<float> m_precision;
    QStringList m_classes;

    QThread* classificationThread;
        ConvolutionalNeuralNetworksClassification* cnnClassification;



};

#endif // CNN_H
