#include "random_forest.h"
#include "ui_random_forest.h"
#include "final.h"

#include <QPixmap>


random_forest::random_forest(QWidget *parent, QString dataPath) :
    QWidget(parent),
    ui(new Ui::random_forest)
{
    ui->setupUi(this);

    this->setWindow();

    classificationThread = new QThread();
    randomForestClassification = new RandomForestClassification();

    randomForestClassification->setPath(dataPath);
    randomForestClassification->moveToThread(classificationThread);
    connect(classificationThread, SIGNAL(started()), this, SLOT(initialization()));

    connect(classificationThread, SIGNAL(started()), randomForestClassification, SLOT(process()));



    connect(randomForestClassification, SIGNAL(donePreprocessing()), this, SLOT(updatePreprocessingFinished()));

    connect(this, SIGNAL(startTrainingClicked()), randomForestClassification, SLOT(train()));
    connect(this, SIGNAL(startTrainingClicked()), this, SLOT(updateStartTraining()));

    connect(randomForestClassification, SIGNAL(trainFinished()), this, SLOT(updateFinished()));
    connect(randomForestClassification, SIGNAL(results(float, float, QList<float>, QList<float>, QStringList)), this, SLOT(setResults(float, float, QList<float>, QList<float>, QStringList)));
    connect(randomForestClassification, SIGNAL(trainFinished()), classificationThread, SLOT(quit()));
    connect(randomForestClassification, SIGNAL(trainFinished()), randomForestClassification, SLOT(deleteLater()));
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

float random_forest::getAccuracy()
{
    return m_testAcc;
}

void random_forest::initialization()
{
    ui->preprocessProgressBar->setMinimum(0);
    ui->preprocessProgressBar->setMaximum(0);

    ui->trainProgressBar->setValue(0);
    ui->startTrainingButton->setEnabled(true);
    ui->showResultsButton->setEnabled(false);
    ui->valueLabel->setVisible(false);
    ui->resultsWidget->setCurrentIndex(4);


//    QPixmap wait("C:/Users/ancab/Desktop");
//    wait = wait.scaled(QSize(500,500), Qt::KeepAspectRatio, Qt::SmoothTransformation);
//    ui->waitLabel->setPixmap(wait);
}

void random_forest::on_startTrainingButton_clicked()
{
    emit startTrainingClicked();
}

void random_forest::updatePreprocessingFinished()
{
    ui->resultsWidget->setCurrentIndex(6);
    ui->preprocessProgressBar->setMaximum(100);
    ui->preprocessProgressBar->setValue(100);
    ui->startTrainingButton->setEnabled(true);
}

void random_forest::updateStartTraining()
{
    ui->resultsWidget->setCurrentIndex(5);
    ui->trainProgressBar->setMinimum(0);
    ui->trainProgressBar->setMaximum(0);
}

void random_forest::updateFinished()
{
    ui->resultsWidget->setCurrentIndex(6);
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
    emit compareClicked(m_classes);
}

void random_forest::on_confusionMatrixButton_clicked()
{
    ui->resultsWidget->setCurrentIndex(2);
    ui->valueLabel->setVisible(false);

    QPixmap confusionMatrix(":/resources/results/confusion_matrix_RF.png");
    confusionMatrix = confusionMatrix.scaled(QSize(700,700), Qt::KeepAspectRatio, Qt::SmoothTransformation);
    ui->confusionMatrixLabel->setPixmap(confusionMatrix);
}

void random_forest::on_RocCurveButton_clicked()
{
    ui->resultsWidget->setCurrentIndex(3);
    ui->valueLabel->setVisible(false);


    QPixmap rocCurve(":/resources/results/ROC_curve_RF.png");
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

void random_forest::plotAccuracy()
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

void random_forest::on_precRecallButton_clicked()
{
    ui->resultsWidget->setCurrentIndex(1);
    ui->valueLabel->setVisible(true);

    ui->valueLabel->setText("Drag cursor over bar to display value");

}

void random_forest::on_accuracyButton_clicked()
{
    ui->resultsWidget->setCurrentIndex(0);
    ui->valueLabel->setVisible(false);

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

void random_forest::showValue(bool status, int index, QBarSet *barSet)
{

    if(status)
    {
        ui->valueLabel->setText(QString::number(barSet->at(index), 'g', 2));
    }
    else
        ui->valueLabel->setText("Drag cursor over bar to display value");
}

void random_forest::printa()
{
   qDebug("a");
}

void random_forest::printb()
{
    qDebug("b");

}

void random_forest::setWindow()
{
    ui->accuracyButton->setEnabled(false);
    ui->precRecallButton->setEnabled(false);
    ui->confusionMatrixButton->setEnabled(false);
    ui->RocCurveButton->setEnabled(false);
}
