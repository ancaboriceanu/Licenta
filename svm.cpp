#include "svm.h"
#include "ui_svm.h"

#include <QThread>
#include <QDebug>


svm::svm(QWidget *parent, QString dataPath) :
    QWidget(parent),
    ui(new Ui::svm)
{
    ui->setupUi(this);

    this->setWindow();

    classificationThread = new QThread;
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
    ui->valueLabel->setVisible(false);

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
    emit compareClicked(m_classes);
}

void svm::on_confusionMatrixButton_clicked()
{
    ui->resultsWidget->setCurrentIndex(2);
    ui->valueLabel->setVisible(false);

    QPixmap confusionMatrix(":/resources/results/confusion_matrix_SVM.png");
    confusionMatrix = confusionMatrix.scaled(QSize(700,700), Qt::KeepAspectRatio, Qt::SmoothTransformation);
    ui->confusionMatrixLabel->setPixmap(confusionMatrix);
}

void svm::on_RocCurveButton_clicked()
{
    ui->resultsWidget->setCurrentIndex(3);
    ui->valueLabel->setVisible(false);

    QPixmap rocCurve(":/resources/results/ROC_curve_SVM.png");
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
    ui->valueLabel->setVisible(false);

}

void svm::on_precRecallButton_clicked()
{
    ui->resultsWidget->setCurrentIndex(1);
    ui->valueLabel->setVisible(true);
    ui->valueLabel->setText("Drag cursor over bar to display value");

}

void svm::plotPrecisionRecall()
{
    QBarSet *set0 = new QBarSet("Precision");
    QBarSet *set1 = new QBarSet("Recall");

    *set0 << m_precision.at(0) << m_precision.at(1) << m_precision.at(2) << m_precision.at(3) << m_precision.at(4) << m_precision.at(5);
    *set1 << m_recall.at(0) << m_recall.at(1) << m_recall.at(2) << m_recall.at(3) << m_recall.at(4) << m_recall.at(5);


    QBarSeries *series = new QBarSeries();
    connect(series, SIGNAL(hovered(bool,int,QBarSet*)), this, SLOT(showValue(bool,int,QBarSet*)));
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
    axisY->setTickCount(11);
    chart->addAxis(axisY, Qt::AlignLeft);
    series->attachAxis(axisY);

    chart->legend()->setVisible(true);
    chart->legend()->setAlignment(Qt::AlignTop);

    QChartView *chartView = new QChartView(chart);
    chartView->setRenderHint(QPainter::Antialiasing);

    QLayoutItem *itemToRemove;
    while ((itemToRemove = ui->precRecallGridLayout->takeAt(0)) != 0)
    {
        delete itemToRemove;
    }

    ui->precRecallGridLayout->addWidget(chartView);
}

void svm::plotAccuracy()
{
    QPieSeries *trainSeries = new QPieSeries();
    trainSeries->append("Train Accuracy", m_trainAcc);
    trainSeries->append("", 1 - m_trainAcc);

    QPieSlice *trainSlice1 = trainSeries->slices().at(0);

    trainSlice1->setBorderWidth(0);
    trainSlice1->setBorderColor(Qt::darkCyan);
    trainSlice1->setBrush(Qt::cyan);

    trainSlice1->setLabelVisible();
    trainSlice1->setLabelBrush(Qt::darkCyan);
    trainSeries->setLabelsPosition(QPieSlice::LabelInsideHorizontal);
    trainSlice1->setLabel(QString("%1%").arg(100*trainSlice1->percentage(), 0, 'f', 1));
    QPieSlice *trainSlice2 = trainSeries->slices().at(1);
    trainSlice2->setBorderWidth(0);

    if(m_trainAcc == 1)
    {
        trainSlice2->setBrush(Qt::cyan);
        trainSlice2->setBorderColor(Qt::cyan);

    }
    else
    {
        trainSlice2->setBrush(Qt::darkCyan);
        trainSlice2->setBorderColor(Qt::darkCyan);

    }

    QPieSeries *testSeries = new QPieSeries();
    testSeries->append("Test Accuracy", m_testAcc);
    testSeries->append("", 1 - m_testAcc);

    QPieSlice *testSlice1 = testSeries->slices().at(0);
    testSlice1->setBorderWidth(0);
    testSlice1->setBorderColor(Qt::darkCyan);
    testSlice1->setBrush(Qt::cyan);
    testSlice1->setLabelVisible();
    testSlice1->setLabelBrush(Qt::darkCyan);
    testSeries->setLabelsPosition(QPieSlice::LabelInsideHorizontal);
    testSlice1->setLabel(QString("%1%").arg(100*testSlice1->percentage(), 0, 'f', 1));
    QPieSlice *testSlice2 = testSeries->slices().at(1);
    testSlice2->setBorderWidth(0);
    testSlice2->setBorderColor(Qt::cyan);

    if(m_testAcc == 1)
    {
        testSlice2->setBrush(Qt::cyan);
        testSlice2->setBorderColor(Qt::cyan);
    }
    else
    {
        testSlice2->setBrush(Qt::darkCyan);
        testSlice2->setBorderColor(Qt::darkCyan);
    }

    QChart *trainChart = new QChart();
    trainChart->addSeries(trainSeries);

    QChart *testChart = new QChart();
    testChart->addSeries(testSeries);

    trainChart->setTitle("Train Accuracy");
    testChart->setTitle("Test Accuracy");

    trainChart->legend()->hide();
    testChart->legend()->hide();

    QChartView *trainChartView = new QChartView(trainChart);
    QChartView *testChartView = new QChartView(testChart);

    trainChartView->setRenderHint(QPainter::Antialiasing);
    testChartView->setRenderHint(QPainter::Antialiasing);

    QLayoutItem *trainItemToRemove;
    QLayoutItem *testItemToRemove;

    while (((testItemToRemove = ui->testGridLayout->takeAt(0)) != 0) && ((trainItemToRemove = ui->trainGridLayout->takeAt(0)) != 0))
    {
        delete testItemToRemove;
        delete trainItemToRemove;
    }

    ui->trainGridLayout->addWidget(trainChartView);
    ui->testGridLayout->addWidget(testChartView);
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

void svm::showValue(bool status, int index, QBarSet *barSet)
{
    if(status)
    {
        ui->valueLabel->setText(QString::number(barSet->at(index), 'g', 2));
    }
    else
        ui->valueLabel->setText("Drag cursor over bar to display value");
}

void svm::setWindow()
{
    ui->accuracyButton->setEnabled(false);
    ui->precRecallButton->setEnabled(false);
    ui->confusionMatrixButton->setEnabled(false);
    ui->RocCurveButton->setEnabled(false);
}
