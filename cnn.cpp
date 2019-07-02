#include "cnn.h"
#include "ui_cnn.h"
#include <QDebug>
#include <QThread>

#include <QtCharts/QPieSeries>
#include <QtCharts/QPieSlice>

#include <QSplineSeries>
#include <QFile>

#include <QtGui/QPainter>
#include <QtGui/QFontMetrics>
#include <QtWidgets/QGraphicsSceneMouseEvent>
#include <QtGui/QMouseEvent>
#include <QtCharts/QChart>

#include <QToolTip>

QT_CHARTS_USE_NAMESPACE

cnn::cnn(QWidget *parent, QString dataPath) :
    QWidget(parent),
    ui(new Ui::cnn)
{
    ui->setupUi(this);

    ui->trainProgressBar->setValue(0);

    classificationThread = new QThread();
    cnnClassification = new ConvolutionalNeuralNetworksClassification();

    cnnClassification->setPath(dataPath);
    cnnClassification->moveToThread(classificationThread);


    connect(this, SIGNAL(startTrainingClicked()), cnnClassification, SLOT(train()));


    connect(this, SIGNAL(startTrainingClicked()), this, SLOT(updateStartTraining()));
   // connect(m_timer, SIGNAL(timeout()), this, SLOT(updatePlot()));


    connect(cnnClassification, SIGNAL(trainFinished()), this, SLOT(updateTrainFinished()));

    connect(this, SIGNAL(startTestingClicked()), cnnClassification, SLOT(test()));
    connect(this, SIGNAL(startTestingClicked()), this, SLOT(initialization()));


    connect(this, SIGNAL(startTestingClicked()), this, SLOT(updateStartTesting()));

    connect(cnnClassification, SIGNAL(testFinished()), this, SLOT(updateTestFinished()));
    connect(cnnClassification, SIGNAL(results(float, QList<float>, QList<float>, QStringList)), this, SLOT(setResults(float, QList<float>, QList<float>, QStringList)));


    connect(cnnClassification, SIGNAL(testFinished()), classificationThread, SLOT(quit()));
    connect(cnnClassification, SIGNAL(testFinished()), cnnClassification, SLOT(deleteLater()));

    connect(classificationThread, SIGNAL(finished()), classificationThread, SLOT(deleteLater()));

    classificationThread->start();
}

cnn::~cnn()
{
    delete ui;
}

void cnn::initialization()
{
    ui->stackedWidget->setVisible(false);
    ui->trainProgressBar->setValue(0);
    //ui->trainProgressBar->setVisible(false);
    ui->showResultsButton->setEnabled(false);
    ui->valueLabel->setVisible(false);

//    QPixmap wait("C:/Users/ancab/Desktop/reload.png");
//    wait = wait.scaled(QSize(500,500), Qt::KeepAspectRatio, Qt::SmoothTransformation);
//    ui->waitLabel->setPixmap(wait);

}


void cnn::updateStartTraining()
{

    m_timer = new QTimer(this);
    connect(m_timer, SIGNAL(timeout()), this, SLOT(updatePlot()));
    m_timer->start(25);

    //ui->trainProgressBar->setVisible(true);
    ui->trainProgressBar->setMinimum(0);
    ui->trainProgressBar->setMaximum(0);
    ui->testRadioButton->setEnabled(false);
    ui->stackedWidget->setVisible(true);
    ui->stackedWidget->setCurrentIndex(0);

}

void cnn::updateStartTesting()
{
    //ui->trainProgressBar->setVisible(true);
    ui->trainProgressBar->setMinimum(0);
    ui->trainProgressBar->setMaximum(0);
    ui->trainRadioButton->setEnabled(false);
    ui->stackedWidget->setCurrentIndex(1);
}

void cnn::updateTrainFinished()
{
    ui->trainProgressBar->setMaximum(100);
    ui->trainProgressBar->setValue(100);
    ui->testRadioButton->setEnabled(true);
}

void cnn::updateTestFinished()
{
    ui->trainProgressBar->setMaximum(100);
    ui->trainProgressBar->setValue(100);
    ui->showResultsButton->setEnabled(true);
    ui->trainRadioButton->setEnabled(true);
}

void cnn::on_backButton_clicked()
{
    emit backClicked();
}

void cnn::on_compareButton_clicked()
{
    emit compareClicked(m_classes);
}

void cnn::on_confusionMatrixButton_clicked()
{
    ui->resultsWidget->setCurrentIndex(2);
    ui->valueLabel->setVisible(false);

    QPixmap confusionMatrix(":/resources/results/confusion_matrix_CNN.png");
    confusionMatrix = confusionMatrix.scaled(QSize(700,700), Qt::KeepAspectRatio, Qt::SmoothTransformation);
    ui->confusionMatrixLabel->setPixmap(confusionMatrix);
}

void cnn::on_RocCurveButton_clicked()
{
    ui->resultsWidget->setCurrentIndex(3);
    ui->valueLabel->setVisible(false);

    QPixmap rocCurve(":/resources/results/ROC_curve_CNN.png");
    rocCurve = rocCurve.scaled(QSize(700,700), Qt::KeepAspectRatio, Qt::SmoothTransformation);
    ui->rocCurveLabel->setPixmap(rocCurve);
}

void cnn::on_showResultsButton_clicked()
{
    ui->stackedWidget->setVisible(true);
    ui->stackedWidget->setCurrentIndex(1);

    this->on_accuracyButton_clicked();

    ui->accuracyButton->setEnabled(true);
    ui->precRecallButton->setEnabled(true);
    ui->confusionMatrixButton->setEnabled(true);
    ui->RocCurveButton->setEnabled(true);

    plotPrecisionRecall();
    plotAccuracy();


}

void cnn::updatePlot()
{
    QSplineSeries *lossSeries = new QSplineSeries();
    QSplineSeries *accSeries = new QSplineSeries();

    QPen lossPen = lossSeries->pen();
    lossPen.setBrush(Qt::red);
    lossPen.setWidth(2);
    lossSeries->setPen(lossPen);

    QPen accPen = accSeries->pen();
    accPen.setBrush(Qt::darkGreen);
    accPen.setWidth(2);
    accSeries->setPen(accPen);

    lossSeries->setName("Loss");
    accSeries->setName("Accuracy");

//    QValueAxis *axisX = new QValueAxis;
//    axisX->setTitleText("Batch / Epoch");
//    axisX->setLabelsVisible(true);

//    QBarCategoryAxis *axisX = new QBarCategoryAxis();
//    QStringList categories;

    float prevEpoch = 0;

//    X2->append("Epoch 2", 602);
//    X2->append("Epoch 3", 903);

    QList<float> line_n;
    QFile inputFile("D:/Licenta/Interfata/resources/results/cnn_plot_values.csv");
    if (inputFile.open(QIODevice::ReadOnly))
    {
        QTextStream in(&inputFile);
        while(!in.atEnd())
        {
            QString line = in.readLine();
            line_n = splitString(line);
            //qDebug() << line_n;
            lossSeries->append(line_n.at(0),line_n.at(1));
            accSeries->append(line_n.at(0),line_n.at(2));

//            if(fmod(line_n.at(0),301.0) == 0)
//            {
//                X2->append(QString("Epoch %1").arg(QString::number(line_n.at(0) / 301) + 1), (line_n.at(0) / 301 + 1));

//            }

            //axisX->setLabelFormat(QString("%1/%2").arg(QString::number(line_n.at(0)), QString::number(line_n.at(3))));

            //qDebug()<<QString(QString("%1/%2").arg(QString::number(line_n.at(0)), QString::number(line_n.at(3))));

            if(line_n.at(3) != prevEpoch)
            {
                prevEpoch = line_n.at(3);
            }
        }
        inputFile.close();
    }

    QCategoryAxis *axisX = new QCategoryAxis;
    axisX->setTitleText("Epochs");

    for(int i = 1; i <= prevEpoch; i++)
    {
        axisX->append(QString("Epoch %1").arg(QString::number(i)), 301*i);
    }

    //axisX->append(categories);

    QChart *chart = new QChart();
    chart->setAcceptHoverEvents(true);
    //chart->createDefaultAxes();
//    chart->addAxis(axisX, Qt::AlignBottom);
//    chart->addAxis(axisY, Qt::AlignBottom);


    chart->addSeries(lossSeries);
    chart->addSeries(accSeries);

    QValueAxis *axisY = new QValueAxis;
    axisY->setRange(0,3);
    axisY->setTickCount(11);
    axisY->setLabelFormat("%.2f");
    axisY->setTitleText("Loss, Accuracy");

    chart->setAxisY(axisY, lossSeries);
    chart->setAxisY(axisY, accSeries);

    chart->setAxisX(axisX, lossSeries);
    chart->setAxisX(axisX, accSeries);

//    chart->setAxisX(axisX, lossSeries);
//    chart->setAxisX(axisX, accSeries);

    chart->setTitle("Loss and Accuracy plot");

    DynamicPlot *chartView = new DynamicPlot(chart);

    chartView->setRenderHint(QPainter::Antialiasing);


    QLayoutItem *itemToRemove;
    while ((itemToRemove = ui->plotGridLayout->takeAt(0)) != 0)
    {
        delete itemToRemove;
    }

    //QToolTip::showText(QPoint(700,700), QString("bjbcjsbc"), chartView, QRect(700,50,700,50),1000);

    ui->plotGridLayout->addWidget(chartView);

}

void cnn::setResults(float testAcc, QList<float> recall, QList<float> precision, QStringList classes)
{
    QList<float> lineNumbers;
    QFile inputFile("D:/Licenta/Interfata/resources/results/cnn_train_acc.csv");
    if (inputFile.open(QIODevice::ReadOnly))
    {
        QTextStream in(&inputFile);
        while(!in.atEnd())
        {
            QString line = in.readLine();
            lineNumbers.append(line.toFloat());
        }
        inputFile.close();
    }

    m_trainAcc = lineNumbers.at(0);
    m_testAcc = testAcc;
    m_recall = recall;
    m_precision = precision;
    m_classes = classes;

}


QList<float> cnn::splitString(QString text)
{
    QList<float> x;
    QStringList list = text.split(QRegExp("[,]"), QString::KeepEmptyParts);

    for (int i = 0; i < list.length(); i++)
        x.append(list.at(i).toFloat());

    return x;
}

void cnn::plotPrecisionRecall()
{
    QBarSet *set0 = new QBarSet("Precision");
    QBarSet *set1 = new QBarSet("Recall");

    *set0 << m_precision.at(0) << m_precision.at(1) << m_precision.at(2) << m_precision.at(3) << m_precision.at(4) << m_precision.at(5);
    *set1 << m_recall.at(0) << m_recall.at(1) << m_recall.at(2) << m_recall.at(3) << m_recall.at(4) << m_recall.at(5);


    QBarSeries *series = new QBarSeries();

//    series->setLabelsVisible(true);
//    set0->setLabelBrush(Qt::darkCyan);
//    set1->setLabelBrush(Qt::darkCyan);
//    series->setLabelsPosition(QBarSeries::LabelsOutsideEnd);

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

void cnn::plotAccuracy()
{
    QPieSeries *trainSeries = new QPieSeries();
    trainSeries->append("Train Accuracy", m_trainAcc);
    trainSeries->append("", 1 - m_trainAcc);

    QPieSlice *trainSlice1 = trainSeries->slices().at(0);

    trainSlice1->setBorderWidth(0);
    trainSlice1->setBorderColor(Qt::darkGreen);
    trainSlice1->setBrush(Qt::green);

    trainSlice1->setLabelVisible();
    trainSlice1->setLabelBrush(Qt::darkGreen);
    trainSeries->setLabelsPosition(QPieSlice::LabelInsideHorizontal);
    trainSlice1->setLabel(QString("%1%").arg(100*trainSlice1->percentage(), 0, 'f', 1));
    QPieSlice *trainSlice2 = trainSeries->slices().at(1);
    trainSlice2->setBorderWidth(0);

    if(m_trainAcc == 1)
    {
        trainSlice2->setBrush(Qt::green);
        trainSlice2->setBorderColor(Qt::green);

    }
    else
    {
        trainSlice2->setBrush(Qt::darkGreen);
        trainSlice2->setBorderColor(Qt::darkGreen);

    }

    QPieSeries *testSeries = new QPieSeries();
    testSeries->append("Test Accuracy", m_testAcc);
    testSeries->append("", 1 - m_testAcc);

    QPieSlice *testSlice1 = testSeries->slices().at(0);
    testSlice1->setBorderWidth(0);
    testSlice1->setBorderColor(Qt::darkGreen);
    testSlice1->setBrush(Qt::green);
    testSlice1->setLabelVisible();
    testSlice1->setLabelBrush(Qt::darkGreen);
    testSeries->setLabelsPosition(QPieSlice::LabelInsideHorizontal);
    testSlice1->setLabel(QString("%1%").arg(100*testSlice1->percentage(), 0, 'f', 1));
    QPieSlice *testSlice2 = testSeries->slices().at(1);
    testSlice2->setBorderWidth(0);
    testSlice2->setBorderColor(Qt::green);

    if(m_testAcc == 1)
    {
        testSlice2->setBrush(Qt::green);
        testSlice2->setBorderColor(Qt::green);
    }
    else
    {
        testSlice2->setBrush(Qt::darkGreen);
        testSlice2->setBorderColor(Qt::darkGreen);
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

void cnn::on_accuracyButton_clicked()
{
    ui->resultsWidget->setCurrentIndex(0);
    ui->valueLabel->setVisible(false);
}

void cnn::on_precRecallButton_clicked()
{
    ui->resultsWidget->setCurrentIndex(1);
    ui->valueLabel->setVisible(true);
    ui->valueLabel->setText("Drag cursor over bar to display value");
}

void cnn::showValue(bool status, int index, QBarSet *barSet)
{
    if(status)
    {
        ui->valueLabel->setText(QString::number(barSet->at(index), 'g', 2));
    }
    else
        ui->valueLabel->setText("Drag cursor over bar to display value");
}

void cnn::on_trainRadioButton_clicked()
{
    emit startTrainingClicked();

}

void cnn::on_testRadioButton_clicked()
{
    emit startTestingClicked();

}
