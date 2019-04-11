#include "randomforestclassification.h"
#include <QtCore>

#include <PythonQt.h>
#include <QDebug>


RandomForestClassification::RandomForestClassification(QObject *parent) : QObject(parent)
{

}

void RandomForestClassification::setPath(QString dataPath)
{
    this->m_dataPath = dataPath;
}

void RandomForestClassification::process()
{
    PythonQt::init(PythonQt::ExternalHelp || PythonQt::RedirectStdOut);

    PythonQtObjectPtr mainContext;
    PythonQtObjectPtr tag;

    mainContext = PythonQt::self()->getMainModule();
    mainContext.evalFile(":/resources/python/classification_random_forest.py");
    tag = mainContext.evalScript("ClassificationRandomForest()\n", Py_eval_input);

    tag.call("set_data_path", QVariantList() << m_dataPath);
    tag.call("set_images_and_labels", QVariantList());
    tag.call("encode_labels", QVariantList());
    tag.call("normalize_data", QVariantList());
    tag.call("split_data", QVariantList());

    m_mainContext = mainContext;
    m_tag = tag;

    emit donePreprocessing();
}

void RandomForestClassification::train()
{
    m_tag.call("train", QVariantList());
    m_tag.call("predict", QVariantList());
    m_tag.call("generate_confusion_matrix", QVariantList());
    m_tag.call("normalize_confusion_matrix", QVariantList());
    m_tag.call("plot_confusion_matrix", QVariantList());
    m_tag.call("plot_ROC_curve", QVariantList());

    QVariant rec = m_tag.call("get_test_recall_per_class", QVariantList());
    QList<float> recall = convertString(rec.toString());

    QVariant prec = m_tag.call("get_test_precision_per_class", QVariantList());
    QList<float> precision = convertString(prec.toString());

    QVariant trainAcc = m_tag.call("get_train_accuracy", QVariantList());
    QVariant testAcc = m_tag.call("get_test_accuracy_oonf_matr", QVariantList());

    QVariant cls = m_tag.call("get_classes", QVariantList());
    qDebug()<<cls;

    QStringList classes = splitString(cls.toString());

    qDebug()<<classes;

    emit randomForestResults(trainAcc.toFloat(), testAcc.toFloat(), recall, precision, classes);

    emit finished();
}

QList<float> RandomForestClassification::convertString(QString string)
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

QStringList RandomForestClassification::splitString(QString string)
{
    string.remove(QRegExp("\\'"));
    string.remove(QRegExp("\\]"));
    string.remove(QRegExp("\\["));

    QStringList x = string.split(QRegExp("[,]"), QString::KeepEmptyParts);

    return x;
}

