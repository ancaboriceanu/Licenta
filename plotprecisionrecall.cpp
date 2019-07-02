#include "plotprecisionrecall.h"

PlotPrecisionRecall::PlotPrecisionRecall()
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
