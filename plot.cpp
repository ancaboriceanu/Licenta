#include "plot.h"


Plot::Plot()
{

}

QChartView* Plot::plotPieChart(float value,QString title)
{
    QPieSeries *series = new QPieSeries();
    series->append(title, value);
    series->append("", 1 - value);

    QPieSlice *slice1 = series->slices().at(0);
    slice1->setBorderWidth(0);
    slice1->setBorderColor(Qt::darkCyan);
    slice1->setBrush(Qt::cyan);
    slice1->setLabelVisible();
    slice1->setLabelBrush(Qt::darkCyan);
    series->setLabelsPosition(QPieSlice::LabelInsideHorizontal);
    slice1->setLabel(QString("%1%").arg(100*slice1->percentage(), 0, 'f', 1));

    QPieSlice *slice2 = series->slices().at(1);
    slice2->setBorderWidth(0);

    if(value == 1)
    {
        slice2->setBrush(Qt::cyan);
        slice2->setBorderColor(Qt::cyan);
    }
    else
    {
        slice2->setBrush(Qt::darkCyan);
        slice2->setBorderColor(Qt::darkCyan);
    }

    QChart *chart = new QChart();
    chart->addSeries(series);
    chart->setTitle(title);
    chart->legend()->hide();
    QChartView *chartView = new QChartView(chart);
    chartView->setRenderHint(QPainter::Antialiasing);

    return chartView;
}

QChartView *Plot::plotBarChart(QList<float> bar1, QList<float> bar2, QStringList classes)
{
    QBarSet *set0 = new QBarSet("Precision");
    QBarSet *set1 = new QBarSet("Recall");

    *set0 << bar2.at(0) << bar2.at(1) << bar2.at(2) << bar2.at(3) << bar2.at(4) << bar2.at(5);
    *set1 << bar1.at(0) << bar1.at(1) << bar1.at(2) << bar1.at(3) << bar1.at(4) << bar1.at(5);

    QBarSeries *series = new QBarSeries();
    //connect(series, SIGNAL(hovered(bool,int,QBarSet*)), this, SLOT(showValue(bool,int,QBarSet*)));
    series->append(set0);
    series->append(set1);

    QChart *chart = new QChart();
    chart->addSeries(series);
    chart->setTitle("Precision and Recall");
    chart->setAnimationOptions(QChart::SeriesAnimations);

    QBarCategoryAxis *axisX = new QBarCategoryAxis();
    QStringList categories;

    categories << classes.at(0) << classes.at(1) << classes.at(2) << classes.at(3) << classes.at(4) << classes.at(5);

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

    return chartView;
}
