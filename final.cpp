#include "final.h"
#include "ui_final.h"

#include <QtWidgets/QApplication>
#include <QtWidgets/QMainWindow>
#include <QtCharts/QChartView>
#include <QtCharts/QLegend>
#include <QtCharts/QBarCategoryAxis>
#include <QtCharts/QLineSeries>
#include <QtCharts/QHorizontalStackedBarSeries>
#include <QtCharts/QCategoryAxis>
#include <QtCharts/QDateTimeAxis>

#include <QtCharts/QPieSeries>
#include <QtCharts/QPieSlice>
#include <QFile>

QT_CHARTS_USE_NAMESPACE



final::final(QWidget *parent, QStringList classes) :
    QWidget(parent),
    ui(new Ui::final)
{
    ui->setupUi(this);

    m_classes = classes;

    ui->backToolButton->setIcon(QIcon("D:/Licenta/Interfata/resources/other_images/left.png"));
    ui->backToolButton->setIconSize(QSize(25,25));
    ui->nextToolButton->setIcon(QIcon("D:/Licenta/Interfata/resources/other_images/right.png"));
    ui->nextToolButton->setIconSize(QSize(25,25));
    ui->homeToolButton->setIcon(QIcon("D:/Licenta/Interfata/resources/other_images/home.png"));
    ui->homeToolButton->setIconSize(QSize(25,25));


    //qDebug()<<classes;
    m_cnnResults.append(this->readFromFile("D:/Licenta/Interfata/resources/results/cnn_train_acc.csv").at(1));
    m_cnnResults.append(this->readFromFile("D:/Licenta/Interfata/resources/results/cnn_results.csv"));

    m_svmResults = this->readFromFile("D:/Licenta/Interfata/resources/results/svm_results.csv");

    m_rfResults = this->readFromFile("D:/Licenta/Interfata/resources/results/random_forest_results.csv");

    this->plotAccuracy();
    this->createTable();
    this->plotTime();
}

final::~final()
{
    delete ui;
}

void final::plotAccuracy()
{
    ui->label_2->setText("Accuracy");
    ui->evaluationWidget->setCurrentIndex(0);

    QPieSeries *cnnSeries = new QPieSeries();
    cnnSeries->append("CNN Test Accuracy", m_cnnResults.at(3).at(0));
    cnnSeries->append("", 1 - m_cnnResults.at(3).at(0));

    QPieSlice *cnnSlice = cnnSeries->slices().at(0);
    cnnSeries->setLabelsVisible();
    cnnSeries->setLabelsPosition(QPieSlice::LabelInsideHorizontal);
    cnnSlice->setLabel(QString("%1%").arg(100*cnnSlice->percentage(), 0, 'f', 1));

    QChart *cnnChart = new QChart();
    cnnChart->legend()->hide();
    cnnChart->addSeries(cnnSeries);
    cnnChart->setTitle("CNN Test Accuracy");

    QChartView *cnnChartView = new QChartView(cnnChart);
    cnnChartView->setRenderHint(QPainter::Antialiasing);

    QPieSeries *svmSeries = new QPieSeries();
    svmSeries->append("SVM Test Accuracy", m_svmResults.at(3).at(0));
    svmSeries->append("", 1 - m_svmResults.at(3).at(0));

    QPieSlice *svmSlice = svmSeries->slices().at(0);
    svmSeries->setLabelsVisible();
    svmSeries->setLabelsPosition(QPieSlice::LabelInsideHorizontal);
    svmSlice->setLabel(QString("%1%").arg(100*svmSlice->percentage(), 0, 'f', 1));

    QChart *svmChart = new QChart();
    svmChart->legend()->hide();
    svmChart->addSeries(svmSeries);
    svmChart->setTitle("SVM Test Accuracy");

    QChartView *svmChartView = new QChartView(svmChart);
    svmChartView->setRenderHint(QPainter::Antialiasing);

    QPieSeries *rfSeries = new QPieSeries();
    rfSeries->append("RF Test Accuracy", m_rfResults.at(3).at(0));
    rfSeries->append("", 1 - m_rfResults.at(3).at(0));

    QPieSlice *rfSlice = rfSeries->slices().at(0);
    rfSeries->setLabelsVisible();
    rfSeries->setLabelsPosition(QPieSlice::LabelInsideHorizontal);
    rfSlice->setLabel(QString("%1%").arg(100*rfSlice->percentage(), 0, 'f', 1));

    QChart *rfChart = new QChart();
    rfChart->legend()->hide();
    rfChart->addSeries(rfSeries);
    rfChart->setTitle("Random Forest Test Accuracy");

    QChartView *rfChartView = new QChartView(rfChart);
    rfChartView->setRenderHint(QPainter::Antialiasing);

    QLayoutItem *cnnItemToRemove;
    QLayoutItem *svmItemToRemove;
    QLayoutItem *rfItemToRemove;

    while (((cnnItemToRemove = ui->cnnGridLayout->takeAt(0)) != 0) && ((svmItemToRemove = ui->svmGridLayout->takeAt(0)) != 0) && ((rfItemToRemove = ui->rfGridLayout->takeAt(0)) != 0))
    {
        delete cnnItemToRemove;
        delete svmItemToRemove;
        delete rfItemToRemove;
    }

    ui->cnnGridLayout->addWidget(cnnChartView);
    ui->svmGridLayout->addWidget(svmChartView);
    ui->rfGridLayout->addWidget(rfChartView);
}

void final::createTable()
{

    ui->metricsTable->setRowCount(9);
    ui->metricsTable->setColumnCount(8);

    ui->metricsTable->verticalHeader()->setVisible(false);
    ui->metricsTable->horizontalHeader()->setVisible(false);
    ui->metricsTable->setShowGrid(true);

    ui->metricsTable->setSpan(1, 1, 2, 1);
//    ui->metricsTable->item(1,1)->setTextAlignment(Qt::AlignCenter);

    ui->metricsTable->setItem(1, 1, new QTableWidgetItem(QString("Class")));

    for(int i = 0; i < m_classes.length(); i++)
    {
        ui->metricsTable->setItem(i+3, 1, new QTableWidgetItem(m_classes.at(i)));
    }

    ui->metricsTable->setItem(1, 2, new QTableWidgetItem(QString("Recall")));
    ui->metricsTable->setItem(1, 5, new QTableWidgetItem(QString("Precision")));
    ui->metricsTable->setSpan(1, 2, 1, 3);
    ui->metricsTable->setSpan(1, 5, 1, 6);

//    ui->metricsTable->item(1,2)->setTextAlignment(Qt::AlignCenter);
//    ui->metricsTable->item(1,5)->setTextAlignment(Qt::AlignCenter);


    ui->metricsTable->setItem(2, 2, new QTableWidgetItem(QString("CNN")));
    ui->metricsTable->setItem(2, 3, new QTableWidgetItem(QString("SVM")));
    ui->metricsTable->setItem(2, 4, new QTableWidgetItem(QString("RF")));

    ui->metricsTable->setItem(2, 5, new QTableWidgetItem(QString("CNN")));
    ui->metricsTable->setItem(2, 6, new QTableWidgetItem(QString("SVM")));
    ui->metricsTable->setItem(2, 7, new QTableWidgetItem(QString("RF")));


    for(int i = 0; i < m_classes.length(); i++)
    {
        ui->metricsTable->setItem(i+3, 2, new QTableWidgetItem(QString::number(m_cnnResults.at(2).at(i), 'f', 2)));
        ui->metricsTable->setItem(i+3, 5, new QTableWidgetItem(QString::number(m_cnnResults.at(3).at(i), 'f', 2)));

        ui->metricsTable->setItem(i+3, 3, new QTableWidgetItem(QString::number(m_svmResults.at(1).at(i), 'f', 2)));
        ui->metricsTable->setItem(i+3, 6, new QTableWidgetItem(QString::number(m_svmResults.at(2).at(i), 'f', 2)));

        ui->metricsTable->setItem(i+3, 4, new QTableWidgetItem(QString::number(m_rfResults.at(1).at(i), 'f', 2)));
        ui->metricsTable->setItem(i+3, 7, new QTableWidgetItem(QString::number(m_rfResults.at(2).at(i),'f', 2)));

    }

//    for(int i = 2; i < ui->metricsTable->rowCount(); i++)
//    {
//        for(int j = 1; j < ui->metricsTable->columnCount(); j++)
//        {
//            if((i == 2) && (j != 1))
//            {
//                ui->metricsTable->item(i, j)->setTextAlignment(Qt::AlignCenter);
//            }
//        }

//    }

    ui->metricsTable->resizeColumnsToContents();
    ui->metricsTable->resizeRowsToContents();
    ui->metricsTable->setRowHeight(0,1);
    ui->metricsTable->setColumnWidth(0, 1);
}

void final::plotTime()
{
    QBarSet *set = new QBarSet("Training Time");

    *set << m_cnnResults.at(0).at(0) / 60 << m_svmResults.at(0).at(0) / 60 << m_rfResults.at(0).at(0) / 60;

    QBarSeries *series = new QBarSeries();
    series->append(set);

    QChart *chart = new QChart();
    chart->addSeries(series);
    chart->setTitle("Test Time");
    chart->setAnimationOptions(QChart::SeriesAnimations);

    QStringList categories;
    categories << "CNN" << "SVM" << "RF";
    QBarCategoryAxis *axisX = new QBarCategoryAxis();
    axisX->setTitleText("Classification Algorithm");
    axisX->append(categories);
    chart->addAxis(axisX, Qt::AlignBottom);
    series->attachAxis(axisX);

    QValueAxis *axisY = new QValueAxis();
    //axisY->setTickCount();
    axisY->setTitleText("Train Time [min]");
    chart->addAxis(axisY, Qt::AlignLeft);
    series->attachAxis(axisY);

    chart->legend()->setVisible(true);
    chart->legend()->setAlignment(Qt::AlignTop);

    QChartView *chartView = new QChartView(chart);
    chartView->setRenderHint(QPainter::Antialiasing);

    QLayoutItem *itemToRemove;
    while ((itemToRemove = ui->timeGridLayout->takeAt(0)) != 0)
    {
        delete itemToRemove;
    }

    ui->timeGridLayout->addWidget(chartView);
}

QList<float> final::splitString(QString text)
{
    QList<float> x;
    QStringList list = text.split(QRegExp("[,]"), QString::KeepEmptyParts);

    for (int i = 0; i < list.length(); i++)
        x.append(list.at(i).toFloat());

    return x;
}

QList<QList<float> > final::readFromFile(QString file)
{
    QList<float> line_n;
    QList<QList<float>> values;
    QFile f(file);
    if (f.open(QIODevice::ReadOnly))
    {
        QTextStream in(&f);
        while(!in.atEnd())
        {
            QString line = in.readLine();
            line_n = splitString(line);
            //qDebug() << line_n;
            values.append(line_n);
        }
        f.close();
    }
    return values;
}

void final::on_nextToolButton_clicked()
{
    if(ui->evaluationWidget->currentIndex() == 0)
    {
        ui->evaluationWidget->setCurrentIndex(1);
        ui->label_2->setText("Metrics");

    }
    else if (ui->evaluationWidget->currentIndex() == 1)
    {
        ui->evaluationWidget->setCurrentIndex(2);
        ui->label_2->setText("Training Time");

    }
}

void final::on_backToolButton_clicked()
{
    if(ui->evaluationWidget->currentIndex() == 1)
    {
        ui->evaluationWidget->setCurrentIndex(0);
        ui->label_2->setText("Accuracy");


    }
    else if (ui->evaluationWidget->currentIndex() == 2)
    {
        ui->evaluationWidget->setCurrentIndex(1);
        ui->label_2->setText("Metrics");

    }
}

void final::on_homeToolButton_clicked()
{
    emit homeClicked();
}
