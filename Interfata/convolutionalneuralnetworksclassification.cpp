#include "convolutionalneuralnetworksclassification.h"

ConvolutionalNeuralNetworksClassification::ConvolutionalNeuralNetworksClassification(QObject *parent) : QObject(parent)
{

}

void ConvolutionalNeuralNetworksClassification::setPath(QString dataPath)
{
    this->m_dataPath = dataPath;
}

void ConvolutionalNeuralNetworksClassification::process()
{
    PythonQt::init(PythonQt::ExternalHelp || PythonQt::RedirectStdOut);

    PythonQtObjectPtr mainContext;
    PythonQtObjectPtr tag;

    mainContext = PythonQt::self()->getMainModule();
    mainContext.evalFile(":/resources/python/classification_cnn.py");
    tag = mainContext.evalScript("ClassificationCNN()\n", Py_eval_input);

    tag.call("set_data_path", QVariantList() << m_dataPath);
    tag.call("create_model", QVariantList());

    m_mainContext = mainContext;
    m_tag = tag;

    emit donePreprocessing();
}

void ConvolutionalNeuralNetworksClassification::train()
{
    m_tag.call("train", QVariantList());
    m_tag.call("get_loss_and_accuracy", QVariantList());
    m_tag.call("plot_loss", QVariantList());
    m_tag.call("plot_accuracy", QVariantList());
    m_tag.call("generate_confusion_matrix", QVariantList());
    m_tag.call("normalize_confusion_matrix", QVariantList());
    m_tag.call("plot_confusion_matrix", QVariantList());
    m_tag.call("plot_ROC_curve", QVariantList());

    emit finished();
}

