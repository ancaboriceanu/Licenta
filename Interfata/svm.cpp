#include "svm.h"
#include "ui_svm.h"

#include <QThread>
#include <QDebug>

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

svm::svm(QWidget *parent, QString dataPath) :
    QWidget(parent),
    ui(new Ui::svm)
{
    ui->setupUi(this);

    this->setWindow();

    QThread* classificationThread = new QThread;
    SupportVectorMachinesClassification* svmClassification = new SupportVectorMachinesClassification();

    svmClassification->setPath(dataPath);
    svmClassification->moveToThread(classificationThread);

    connect(classificationThread, SIGNAL(started()), this, SLOT(initialization()));
    connect(classificationThread, SIGNAL(started()), svmClassification, SLOT(process()));

    connect(svmClassification, SIGNAL(donePreprocessing()), this, SLOT(updatePreprocessingFinished()));

    connect(this, SIGNAL(startTrainingClicked()), svmClassification, SLOT(train()));
    connect(this, SIGNAL(startTrainingClicked()), this, SLOT(updateStartTraining()));

    connect(svmClassification, SIGNAL(finished()), this, SLOT(updateFinished()));
    connect(svmClassification, SIGNAL(svmResults(float, float, QList<float>, QList<float>, QStringList)), this, SLOT(setResults(float, float, QList<float>, QList<float>, QStringList)));

    connect(svmClassification, SIGNAL(finished()), classificationThread, SLOT(quit()));
    connect(svmClassification, SIGNAL(finished()), svmClassification, SLOT(deleteLater()));
    connect(classificationThread, SIGNAL(finished()), classificationThread, SLOT(deleteLater()));

    classificationThread->start();
}

svm::~svm()
{
    delete ui;
}

void svm::initialization()
{
    ui->preprocessProgressBar->setMinimum(0);
    ui->preprocessProgressBar->setMaximum(0);

    ui->trainProgressBar->setValue(0);
    ui->startTrainingButton->setEnabled(false);
    ui->showResultsButton->setEnabled(false);
}

void svm::updatePreprocessingFinished()
{
    ui->preprocessProgressBar->setMaximum(100);
    ui->preprocessProgressBar->setValue(100);
    ui->startTrainingButton->setEnabled(true);
}

void svm::updateStartTraining()
{
    ui->trainProgressBar->setMinimum(0);
    ui->trainProgressBar->setMaximum(0);
}

void svm::updateFinished()
{
    ui->trainProgressBar->setMaximum(100);
    ui->trainProgressBar->setValue(100);
    ui->showResultsButton->setEnabled(true);
}

void svm::on_startTrainingButton_clicked()
{
    emit startTrainingClicked();
}

void svm::on_backButton_clicked()
{
    emit backClicked();
}

void svm::on_compareButton_clicked()
{
    emit compareClicked();
}

void svm::on_confusionMatrixButton_clicked()
{
    ui->resultsWidget->setCurrentIndex(2);

    QPixmap confusionMatrix("D:/Licenta/Interfata/resources/images/confusion_matrix_SVM.png");
    confusionMatrix = confusionMatrix.scaled(QSize(700,700), Qt::KeepAspectRatio, Qt::SmoothTransformation);
    ui->confusionMatrixLabel->setPixmap(confusionMatrix);
}

void svm::on_RocCurveButton_clicked()
{
    ui->resultsWidget->setCurrentIndex(3);

    QPixmap rocCurve("D:/Licenta/Interfata/resources/images/ROC_curve_SVM.png");
    rocCurve = rocCurve.scaled(QSize(700,700), Qt::KeepAspectRatio, Qt::SmoothTransformation);
    ui->rocCurveLabel->setPixmap(rocCurve);
}

void svm::setResults(float trainAcc, float testAcc, QList<float> recall, QList<float> precision, QStringList classes)
{
    m_trainAcc = trainAcc;
    m_testAcc = testAcc;
    m_recall = recall;
    m_precision = precision;
    m_classes = classes;
}

void svm::on_accuracyButton_clicked()
{
    ui->resultsWidget->setCurrentIndex(0);
}

void svm::on_precRecallButton_clicked()
{
    ui->resultsWidget->setCurrentIndex(1);
}

void svm::plotPrecisionRecall()
{
    QBarSet *set0 = new QBarSet("Precision");
    QBarSet *set1 = new QBarSet("Recall");

    *set0 << m_precision.at(0) << m_precision.at(1) << m_precision.at(2) << m_precision.at(3) << m_precision.at(4) << m_precision.at(5);
    *set1 << m_recall.at(0) << m_recall.at(1) << m_recall.at(2) << m_recall.at(3) << m_recall.at(4) << m_recall.at(5);


    QBarSeries *series = new QBarSeries();
    series->append(set0);
    series->append(set1);

    QChart *chart = new QChart();
    chart->addSeries(series);
    chart->setTitle("Precision and Recall");
    chart->setAnimationOptions(QChart::SeriesAnimations);


    QBarCategoryAxis *axisX = new QBarCategoryAxis();
    QStringList categories;

    categories << m_classes.at(0) << m_classes.at(1) << m_classes.at(2) << m_classes.at(3) << m_classes.at(4) << m_classes.at(5);

    axisX->append(categories);
    chart->addAxis(axisX, Qt::AlignBottom);
    series->attachAxis(axisX);

    QValueAxis *axisY = new QValueAxis();
    axisY->setRange(0,1);
    axisY->setTickCount(0.1);
    chart->addAxis(axisY, Qt::AlignLeft);
    series->attachAxis(axisY);

    chart->legend()->setVisible(true);
    chart->legend()->setAlignment(Qt::AlignTop);

    QChartView *chartView = new QChartView(chart);
    chartView->setRenderHint(QPainter::Antialiasing);

    ui->precRecallGridLayout->addWidget(chartView);
}

void svm::plotAccuracy()
{
    QPieSeries *series = new QPieSeries();
    series->append("Train Accuracy", m_trainAcc);
    series->append("Test Accuracy", m_testAcc);

    QPieSlice *slice = series->slices().at(1);
    //slice->setExploded();
    //slice->setLabelVisible();
    slice->setPen(QPen(Qt::darkGreen, 2));
    slice->setBrush(Qt::green);

    QChart *chart = new QChart();
    chart->addSeries(series);
    chart->setTitle("Train and Test Accuracy");
    //chart->legend()->hide();

    QChartView *chartView = new QChartView(chart);
    chartView->setRenderHint(QPainter::Antialiasing);

    ui->accGridLayout->addWidget(chartView);
}

void svm::on_showResultsButton_clicked()
{
    this->on_accuracyButton_clicked();

    ui->accuracyButton->setEnabled(true);
    ui->precRecallButton->setEnabled(true);
    ui->confusionMatrixButton->setEnabled(true);
    ui->RocCurveButton->setEnabled(true);

    plotPrecisionRecall();
    plotAccuracy();
}

void svm::setWindow()
{
    ui->accuracyButton->setEnabled(false);
    ui->precRecallButton->setEnabled(false);
    ui->confusionMatrixButton->setEnabled(false);
    ui->RocCurveButton->setEnabled(false);
}
