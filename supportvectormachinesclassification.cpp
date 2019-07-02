#include "supportvectormachinesclassification.h"
#include "QDir"
#include "formatstring.h"
SupportVectorMachinesClassification::SupportVectorMachinesClassification(QObject *parent) : QObject(parent)
{

}

void SupportVectorMachinesClassification::setPath(QString dataPath)
{
    this->m_dataPath = dataPath;
}

void SupportVectorMachinesClassification::process()
{
    PythonQt::init(PythonQt::ExternalHelp || PythonQt::RedirectStdOut);

    PythonQtObjectPtr mainContext;
    PythonQtObjectPtr tag;

    mainContext = PythonQt::self()->getMainModule();
    mainContext.evalFile(":/resources/python/classification_svm.py");
    tag = mainContext.evalScript("ClassificationSVM()\n", Py_eval_input);
    tag.call("set_data_path", QVariantList() << m_dataPath);
    tag.call("set_images_and_labels", QVariantList());

    tag.call("encode_labels", QVariantList());
    tag.call("reshape_and_normalize_data", QVariantList());
    tag.call("split_data", QVariantList());

    m_mainContext = mainContext;
    m_tag = tag;

    emit donePreprocessing();
}

void SupportVectorMachinesClassification::train()
{
    QDir dir;
    dir.remove("D:/Licenta/Interfata/resources/results/svm_results.csv");

    FormatString* f = new FormatString();

    m_tag.call("train", QVariantList());
    m_tag.call("predict", QVariantList());
    m_tag.call("generate_confusion_matrix", QVariantList());
    m_tag.call("normalize_confusion_matrix", QVariantList());
    m_tag.call("plot_confusion_matrix", QVariantList());
    m_tag.call("plot_ROC_curve", QVariantList());

    QVariant rec = m_tag.call("get_test_recall_per_class", QVariantList());
    QList<float> recall = f->convertString(rec.toString());

    QVariant prec = m_tag.call("get_test_precision_per_class", QVariantList());
    QList<float> precision = f->convertString(prec.toString());

    QVariant trainAcc = m_tag.call("get_train_accuracy", QVariantList());
    QVariant testAcc = m_tag.call("get_test_accuracy_conf_matr", QVariantList());

    QVariant cls = m_tag.call("get_classes", QVariantList());
    QStringList classes = f->splitString(cls.toString());

    qDebug()<<classes;

    emit results(trainAcc.toFloat(), testAcc.toFloat(), recall, precision, classes);

    emit trainFinished();
}

