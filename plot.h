#ifndef PLOT_H
#define PLOT_H

#include <QObject>
#include <QWidget>
#include <QChartView>
#include <QPixmap>
#include <QLayout>

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

class Plot
{
public:
    Plot();

public:
    QChartView *plotPieChart(float value, QString title);
    QChartView *plotBarChart(QList<float> bar1, QList<float> bar2, QStringList classes);
};

#endif // PLOT_H
