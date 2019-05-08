#include "dynamicplot.h"
#include <QDebug>
#include <QToolTip>
#include <QLabel>

DynamicPlot::DynamicPlot(QChart *chart, QWidget *parent) : QChartView(chart, parent)
{


}

void DynamicPlot::mousePressEvent(QMouseEvent *event)
{
    QChartView::setMouseTracking(true);
    QChartView::mousePressEvent(event);
    auto const widgetPos = event->localPos();
//    auto const scenePos = mapToScene(QPoint(static_cast<int>(widgetPos.x()), static_cast<int>(widgetPos.y())));
//    auto const chartItemPos = chart()->mapFromScene(scenePos);
    auto const valueGivenSeries = chart()->mapToValue(widgetPos);

    qDebug()<<widgetPos;
////    qDebug()<<scenePos;
////    qDebug()<<chartItemPos;
    qDebug()<<valueGivenSeries;
    QToolTip::showText(mapToGlobal(event->globalPos()), QString::number(qRound(valueGivenSeries.x())) + ", "
                      + QString::number(valueGivenSeries.y()));

    qDebug()<<mapToGlobal(event->globalPos());

    qDebug()<<QToolTip::text();
}

