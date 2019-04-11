#ifndef SVM_H
#define SVM_H

#include "supportvectormachinesclassification.h"
#include <QThread>
#include <QWidget>

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
    void compareClicked();
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


};

#endif // SVM_H
