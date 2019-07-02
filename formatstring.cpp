#include "formatstring.h"

FormatString::FormatString()
{

}

QList<QList<float> > FormatString::readFromFile(QString file)
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
            line_n = this->convertString(line);
            values.append(line_n);
        }
        f.close();
    }
    return values;
}



QList<float> FormatString::splitStringCnnPlot(QString text)
{
    QList<float> x;
    QStringList list = text.split(QRegExp("[,]"), QString::KeepEmptyParts);

    for (int i = 0; i < list.length(); i++)
        x.append(list.at(i).toFloat());

    return x;
}

QList<float> FormatString::convertString(QString string)
{
    string.remove(QRegExp("\\,"));
    string.remove(QRegExp("\\'"));
    string.remove(QRegExp("\\]"));
    string.remove(QRegExp("\\["));

    QList<float> floatList;

    QStringList x = string.split(QRegExp("[ ]"), QString::KeepEmptyParts);

    for (int i = 0; i < x.length(); i++)
        floatList.append(x.at(i).toFloat());

    return floatList;
}

QStringList FormatString::splitString(QString string)
{
    string.remove(QRegExp("\\'"));
    string.remove(QRegExp("\\]"));
    string.remove(QRegExp("\\["));

    QStringList x = string.split(QRegExp("[,]"), QString::KeepEmptyParts);

    return x;
}
