#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

#include "start.h"
#include "cnn.h"
#include "svm.h"
#include "random_forest.h"
#include "final.h"


namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void goToCnn(QString dataPath);
    void goToSvm(QString dataPath);
    void goToRf(QString dataPath);
    void goBack();
    void compare(QStringList classes);
    void createNew();

private:
    Ui::MainWindow *ui;

};

#endif // MAINWINDOW_H
