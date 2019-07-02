#ifndef ALGORITHMWIDGET_H
#define ALGORITHMWIDGET_H
#include <QThread>


class AlgorithmWidget
{
public:
    AlgorithmWidget();

public:
    virtual ~AlgorithmWidget(){}


public slots:
    virtual void initialization() = 0;
    virtual void updateStartTraining() = 0;
    virtual void updateTrainFinished() = 0;
    virtual void on_backButton_clicked() = 0;
    virtual void on_compareButton_clicked() = 0;
    virtual void on_confusionMatrixButton_clicked() = 0;
    virtual void on_RocCurveButton_clicked() = 0;
    virtual void on_precRecallButton_clicked() = 0;
    virtual void on_accuracyButton_clicked() = 0;
    virtual void on_showResultsButton_clicked() = 0;

private:
    float m_trainAcc;
    float m_testAcc;
    QList<float> m_recall;
    QList<float> m_precision;
    QStringList m_classes;
    QThread* classificationThread;
};

Q_DECLARE_INTERFACE(AlgorithmWidget, "AlgorithmWidget")
#endif // ALGORITHMWIDGET_H
