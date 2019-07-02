#include "convolutionalneuralnetworksclassification.h"
#include "formatstring.h"
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

    m_mainContext = PythonQt::self()->getMainModule();
    m_mainContext.evalFile(":/resources/python/classification_cnn.py");
    m_tag = m_mainContext.evalScript("ClassificationCNN()\n", Py_eval_input);

    m_tag.call("set_data_path", QVariantList() << m_dataPath);
    m_tag.call("set_generators", QVariantList());
    m_tag.call("create_model", QVariantList());

    m_tag.call("train", QVariantList());

    emit trainFinished();
}

void ConvolutionalNeuralNetworksClassification::test()
{
    QDir dir;
    if (dir.exists("D:/Licenta/Interfata/resources/results/cnn_results.csv"))
    {
        dir.remove("D:/Licenta/Interfata/resources/results/cnn_results.csv");
    }

    PythonQt::init(PythonQt::ExternalHelp || PythonQt::RedirectStdOut);

    m_mainContext = PythonQt::self()->getMainModule();
    m_mainContext.evalFile(":/resources/python/classification_cnn.py");
    m_tag = m_mainContext.evalScript("ClassificationCNN()\n", Py_eval_input);
    FormatString* f = new FormatString();
    m_tag.call("set_data_path", QVariantList() << m_dataPath);

    m_tag.call("set_generators", QVariantList());

    m_tag.call("create_model", QVariantList());


    m_tag.call("load", QVariantList());

    m_tag.call("get_test_accuracy", QVariantList());

    m_tag.call("generate_confusion_matrix", QVariantList());
    m_tag.call("normalize_confusion_matrix", QVariantList());
    m_tag.call("plot_confusion_matrix", QVariantList());
    m_tag.call("plot_ROC_curve", QVariantList());

    QVariant rec = m_tag.call("get_test_recall_per_class", QVariantList());
    QList<float> recall = f->convertString(rec.toString());

    QVariant prec = m_tag.call("get_test_precision_per_class", QVariantList());
    QList<float> precision = f->convertString(prec.toString());


    //QVariant trainAcc = tag.call("get_train_accuracy", QVariantList());
    //qDebug()<<trainAcc;

    QVariant testAcc = m_tag.call("get_test_accuracy_conf_matr", QVariantList());

    QVariant cls = m_tag.call("get_classes", QVariantList());

    QStringList classes = f->splitString(cls.toString());

    emit results(testAcc.toFloat(), recall, precision, classes);

    emit testFinished();
}

