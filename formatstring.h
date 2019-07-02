#ifndef FORMATSTRING_H
#define FORMATSTRING_H

#include <QObject>
#include <QFile>
 #include <QTextStream>

class FormatString
{
public:
    FormatString();

public:
    QList<QList<float> > readFromFile(QString file);
    QList<float> splitStringCnnPlot(QString text);
    QList<float> convertString(QString string);
    QStringList splitString(QString string);
};

#endif // FORMATSTRING_H
