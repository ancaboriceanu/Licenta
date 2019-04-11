#include "random_forest.h"
#include "ui_random_forest.h"

#include <QPixmap>

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

#include <QtCharts/QPieSeries>
#include <QtCharts/QPieSlice>

QT_CHARTS_USE_NAMESPACE

random_forest::random_forest(QWidget *parent, QString dataPath) :
    QWidget(parent),
    ui(new Ui::random_forest)
{
    ui->setupUi(this);

    this->setWindow();

    QThread* classificationThread = new QThread;
    RandomForestClassification* randomForestClassification = new RandomForestClassification();

    randomForestClassification->setPath(dataPath);
    randomForestClassification->moveToThread(classificationThread);

    connect(classificationThread, SIGNAL(started()), this, SLOT(initialization()));
    connect(classificationThread, SIGNAL(started()), randomForestClassification, SLOT(process()));

    connect(randomForestClassification, SIGNAL(donePreprocessing()), this, SLOT(updatePreprocessingFinished()));

    connect(this, SIGNAL(startTrainingClicked()), randomForestClassification, SLOT(train()));
    connect(this, SIGNAL(startTrainingClicked()), this, SLOT(updateStartTraining()));

    connect(randomForestClassification, SIGNAL(finished()), this, SLOT(updateFinished()));
    connect(randomForestClassification, SIGNAL(randomForestResults(float, float, QList<float>, QList<float>, QStringList)), this, SLOT(setResults(float, float, QList<float>, QList<float>, QStringList)));
    connect(randomForestClassification, SIGNAL(finished()), classificationThread, SLOT(quit()));
    connect(randomForestClassification, SIGNAL(finished()), randomForestClassification, SLOT(deleteLater()));
    connect(classificationThread, SIGNAL(finished()), classificationThread, SLOT(deleteLater()));

    classificationThread->start();

    /*
    QPixmap button("C:/Users/ancab/Desktop/back.png");
    button = button.scaled(QSize(30,30), Qt::KeepAspectRatio, Qt::SmoothTransformation);
    ui->backButtonLabel->setPixmap(button);*/
}

random_forest::~random_forest()
{
    delete ui;
}

void random_forest::initialization()
{
    ui->preprocessProgressBar->setMinimum(0);
    ui->preprocessProgressBar->setMaximum(0);

    ui->trainProgressBar->setValue(0);
    ui->startTrainingButton->setEnabled(false);
    ui->showResultsButton->setEnabled(false);
}

void random_forest::on_startTrainingButton_clicked()
{
    emit startTrainingClicked();
}

void random_forest::updatePreprocessingFinished()
{
    ui->preprocessProgressBar->setMaximum(100);
    ui->preprocessProgressBar->setValue(100);
    ui->startTrainingButton->setEnabled(true);
}

void random_forest::updateStartTraining()
{
    ui->trainProgressBar->setMinimum(0);
    ui->trainProgressBar->setMaximum(0);
}

void random_forest::updateFinished()
{
    ui->trainProgressBar->setMaximum(100);
    ui->trainProgressBar->setValue(100);
    ui->showResultsButton->setEnabled(true);
}

void random_forest::on_backButton_clicked()
{
    emit backClicked();
}

void random_forest::on_compareButton_clicked()
{
    emit compareClicked();
}

void random_forest::on_confusionMatrixButton_clicked()
{
    ui->resultsWidget->setCurrentIndex(2);

    QPixmap confusionMatrix("D:/Licenta/Interfata/resources/images/confusion_matrix_RF.png");
    confusionMatrix = confusionMatrix.scaled(QSize(700,700), Qt::KeepAspectRatio, Qt::SmoothTransformation);
    ui->confusionMatrixLabel->setPixmap(confusionMatrix);
}

void random_forest::on_RocCurveButton_clicked()
{
    ui->resultsWidget->setCurrentIndex(3);

    QPixmap rocCurve("D:/Licenta/Interfata/resources/images/ROC_curve_RF.png");
    rocCurve = rocCurve.scaled(QSize(700,700), Qt::KeepAspectRatio, Qt::SmoothTransformation);
    ui->rocCurveLabel->setPixmap(rocCurve);
}

void random_forest::setResults(float trainAcc, float testAcc, QList<float> recall, QList<float> precision, QStringList classes)
{
    m_trainAcc = trainAcc;
    m_testAcc = testAcc;
    m_recall = recall;
    m_precision = precision;
    m_classes = classes;
}

void random_forest::plotPrecisionRecall()
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

void random_forest::plotAccuracy()
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

void random_forest::on_precRecallButton_clicked()
{
    ui->resultsWidget->setCurrentIndex(1);
}

void random_forest::on_accuracyButton_clicked()
{
    ui->resultsWidget->setCurrentIndex(0);
}

void random_forest::on_showResultsButton_clicked()
{
    this->on_accuracyButton_clicked();

    ui->accuracyButton->setEnabled(true);
    ui->precRecallButton->setEnabled(true);
    ui->confusionMatrixButton->setEnabled(true);
    ui->RocCurveButton->setEnabled(true);

    plotPrecisionRecall();
    plotAccuracy();
}

void random_forest::setWindow()
{
    ui->accuracyButton->setEnabled(false);
    ui->precRecallButton->setEnabled(false);
    ui->confusionMatrixButton->setEnabled(false);
    ui->RocCurveButton->setEnabled(false);
}
