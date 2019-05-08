#ifndef DYNAMICPLOT_H
#define DYNAMICPLOT_H

#include <QChartView>

QT_CHARTS_USE_NAMESPACE

class DynamicPlot : public QChartView
{
public:
    DynamicPlot(QChart *chart, QWidget *parent = 0);

public slots:
    void mousePressEvent(QMouseEvent *event);
};

#endif // DYNAMICPLOT_H
