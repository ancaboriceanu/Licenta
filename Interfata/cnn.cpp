#include "cnn.h"
#include "ui_cnn.h"
#include <QDebug>

cnn::cnn(QWidget *parent, QString dataPath) :
    QWidget(parent),
    ui(new Ui::cnn)
{
    ui->setupUi(this);

    this->setWindow();

    QThread* classificationThread = new QThread;
    ConvolutionalNeuralNetworksClassification* cnnClassification = new ConvolutionalNeuralNetworksClassification();

    cnnClassification->setPath(dataPath);
    cnnClassification->moveToThread(classificationThread);

    connect(classificationThread, SIGNAL(started()), this, SLOT(initialization()));
    connect(classificationThread, SIGNAL(started()), cnnClassification, SLOT(process()));

    connect(cnnClassification, SIGNAL(donePreprocessing()), this, SLOT(updatePreprocessingFinished()));

    connect(this, SIGNAL(startTrainingClicked()), cnnClassification, SLOT(train()));
    connect(this, SIGNAL(startTrainingClicked()), this, SLOT(updateStartTraining()));

    connect(cnnClassification, SIGNAL(finished()), this, SLOT(updateFinished()));
    connect(cnnClassification, SIGNAL(finished()), classificationThread, SLOT(quit()));
    connect(cnnClassification, SIGNAL(finished()), cnnClassification, SLOT(deleteLater()));
    connect(classificationThread, SIGNAL(finished()), classificationThread, SLOT(deleteLater()));

    classificationThread->start();
}

cnn::~cnn()
{
    delete ui;
}

void cnn::initialization()
{
    ui->preprocessProgressBar->setMinimum(0);
    ui->preprocessProgressBar->setMaximum(0);

    ui->trainProgressBar->setValue(0);
    ui->startTrainingButton->setEnabled(false);
    ui->showResultsButton->setEnabled(false);
}

void cnn::on_startTrainingButton_clicked()
{
    emit startTrainingClicked();
}

void cnn::updatePreprocessingFinished()
{
    ui->preprocessProgressBar->setMaximum(100);
    ui->preprocessProgressBar->setValue(100);
    ui->startTrainingButton->setEnabled(true);
}

void cnn::updateStartTraining()
{
    ui->trainProgressBar->setMinimum(0);
    ui->trainProgressBar->setMaximum(0);
}

void cnn::updateFinished()
{
    ui->trainProgressBar->setMaximum(100);
    ui->trainProgressBar->setValue(100);
    ui->showResultsButton->setEnabled(true);
}

void cnn::on_backButton_clicked()
{
    emit backClicked();
}

void cnn::on_compareButton_clicked()
{
    emit compareClicked();
}

void cnn::on_confusionMatrixButton_clicked()
{
    ui->resultsWidget->setCurrentIndex(2);

    QPixmap confusionMatrix("D:/Licenta/Interfata/resources/images/confusion_matrix_CNN.png");
    confusionMatrix = confusionMatrix.scaled(QSize(700,700), Qt::KeepAspectRatio, Qt::SmoothTransformation);
    ui->confusionMatrixLabel->setPixmap(confusionMatrix);
}

void cnn::on_RocCurveButton_clicked()
{
    ui->resultsWidget->setCurrentIndex(3);

    QPixmap rocCurve("D:/Licenta/Interfata/resources/images/ROC_curve_CNN.png");
    rocCurve = rocCurve.scaled(QSize(700,700), Qt::KeepAspectRatio, Qt::SmoothTransformation);
    ui->rocCurveLabel->setPixmap(rocCurve);
}

void cnn::on_showResultsButton_clicked()
{
    ui->accuracyButton->setEnabled(true);
    ui->precRecallButton->setEnabled(true);
    ui->confusionMatrixButton->setEnabled(true);
    ui->RocCurveButton->setEnabled(true);

}

void cnn::setWindow()
{
    ui->accuracyButton->setEnabled(false);
    ui->precRecallButton->setEnabled(false);
    ui->confusionMatrixButton->setEnabled(false);
    ui->RocCurveButton->setEnabled(false);
}
