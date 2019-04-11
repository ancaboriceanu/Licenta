#ifndef CNN_H
#define CNN_H

#include "convolutionalneuralnetworksclassification.h"

#include <QWidget>

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

    void on_showResultsButton_clicked();


private:
    void setWindow();

private:
    Ui::cnn *ui;
};

#endif // CNN_H
