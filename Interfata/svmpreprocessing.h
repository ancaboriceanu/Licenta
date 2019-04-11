#ifndef SVMPREPROCESSING_H
#define SVMPREPROCESSING_H
#include <QObject>
#include <PythonQt.h>


class SvmPreprocessing : public QObject
{
    Q_OBJECT

public:
    SvmPreprocessing();
    void set_path(QString data_path);

public slots:
    void preprocess();
    void train();

signals:
    void result(float train_acc, float test_acc);
    void conf(QList<QList<int>> confmatrix, QStringList classes);
    void finished();
    void donePreprocessing();

private:
    void get_confusion_matrix(QString conf_matrix);
    void get_classes(QString all_classes);

private:
    QString data_path;
    QList<QList<int>> confusion_matrix;
    QList<int> line;
    QStringList classes;
    PythonQtObjectPtr mainContext;
    PythonQtObjectPtr tag;



};

#endif // SVMPREPROCESSING_H
