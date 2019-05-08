#include "convolutionalneuralnetworksclassification.h"

ConvolutionalNeuralNetworksClassification::ConvolutionalNeuralNetworksClassification(QObject *parent) : QObject(parent)
{

}

void ConvolutionalNeuralNetworksClassification::setPath(QString dataPath)
{
    this->m_dataPath = dataPath;
}

void ConvolutionalNeuralNetworksClassification::train()
{
    QDir dir;

    if (dir.exists("D:/Licenta/Interfata/resources/results/cnn_plot_values.csv"))
    {
        dir.remove("D:/Licenta/Interfata/resources/results/cnn_plot_values.csv");
    }

    if (dir.exists("D:/Licenta/Interfata/resources/results/cnn_train_acc.csv"))
    {
        dir.remove("D:/Licenta/Interfata/resources/results/cnn_train_acc.csv");
    }

    PythonQt::init(PythonQt::ExternalHelp || PythonQt::RedirectStdOut);

    PythonQtObjectPtr mainContext;
    PythonQtObjectPtr tag;

    mainContext = PythonQt::self()->getMainModule();
    mainContext.evalFile(":/resources/python/classification_cnn.py");
    tag = mainContext.evalScript("ClassificationCNN()\n", Py_eval_input);

    tag.call("set_data_path", QVariantList() << m_dataPath);
    tag.call("set_generators", QVariantList());
    tag.call("create_model", QVariantList());

    tag.call("train", QVariantList());

    emit trainFinished();
}

void ConvolutionalNeuralNetworksClassification::test()
{
//    QDir dir;
//    if (dir.exists("D:/Licenta/Interfata/resources/results/cnn_results.csv"))
//    {
//        dir.remove("D:/Licenta/Interfata/resources/results/cnn_results.csv");
//    }

    PythonQt::init(PythonQt::ExternalHelp || PythonQt::RedirectStdOut);

    PythonQtObjectPtr mainContext;
    PythonQtObjectPtr tag;

    mainContext = PythonQt::self()->getMainModule();
    mainContext.evalFile(":/resources/python/classification_cnn.py");
    tag = mainContext.evalScript("ClassificationCNN()\n", Py_eval_input);

    tag.call("set_data_path", QVariantList() << m_dataPath);

    tag.call("set_generators", QVariantList());

    tag.call("create_model", QVariantList());


    tag.call("load", QVariantList());
    tag.call("get_test_accuracy", QVariantList());

    tag.call("generate_confusion_matrix", QVariantList());
    tag.call("normalize_confusion_matrix", QVariantList());
    tag.call("plot_confusion_matrix", QVariantList());
    tag.call("plot_ROC_curve", QVariantList());

    QVariant rec = tag.call("get_test_recall_per_class", QVariantList());
    QList<float> recall = convertString(rec.toString());

    QVariant prec = tag.call("get_test_precision_per_class", QVariantList());
    QList<float> precision = convertString(prec.toString());


    //QVariant trainAcc = tag.call("get_train_accuracy", QVariantList());
    //qDebug()<<trainAcc;

    QVariant testAcc = tag.call("get_test_accuracy_conf_matr", QVariantList());

    QVariant cls = tag.call("get_classes", QVariantList());

    QStringList classes = splitString(cls.toString());

    emit cnnResults(testAcc.toFloat(), recall, precision, classes);

    emit testFinished();
}

QList<float> ConvolutionalNeuralNetworksClassification::convertString(QString string)
{
    string.remove(QRegExp("\\,"));
    string.remove(QRegExp("\\'"));
    string.remove(QRegExp("\\]"));
    string.remove(QRegExp("\\["));

    QList<float> floatList;

    QStringList x = string.split(QRegExp("[ ]"), QString::SkipEmptyParts);

    for (int i = 0; i < x.length(); i++)
        floatList.append(x.at(i).toFloat());

    return floatList;
}

QStringList ConvolutionalNeuralNetworksClassification::splitString(QString string)
{
    string.remove(QRegExp("\\'"));
    string.remove(QRegExp("\\]"));
    string.remove(QRegExp("\\["));

    QStringList x = string.split(QRegExp("[,]"), QString::SkipEmptyParts);

    return x;
}

