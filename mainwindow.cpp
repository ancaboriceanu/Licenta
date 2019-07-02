#include "mainwindow.h"
#include "ui_mainwindow.h"


MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    start *startPage = new start(this);
    ui->stackedWidget->insertWidget(0, startPage);
    ui->stackedWidget->setCurrentIndex(0);


    connect(startPage, SIGNAL(cnnNextClicked(QString)), this, SLOT(goToCnn(QString)));
    connect(startPage, SIGNAL(svmNextClicked(QString)), this, SLOT(goToSvm(QString)));
    connect(startPage, SIGNAL(rfNextClicked(QString)), this, SLOT(goToRf(QString)));

}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::goToCnn(QString dataPath)
{
    cnn *CNN = new cnn(this, dataPath);
    ui->stackedWidget->insertWidget(1, CNN);

    connect(CNN, SIGNAL(backClicked()), this, SLOT(goBack()));
    connect(CNN, SIGNAL(compareClicked(QStringList)), this, SLOT(compare(QStringList)));
    ui->stackedWidget->setCurrentIndex(1);
}

void MainWindow::goToSvm(QString dataPath)
{
    svm *SVM = new svm(this, dataPath);
    ui->stackedWidget->insertWidget(2, SVM);

    connect(SVM, SIGNAL(backClicked()), this, SLOT(goBack()));
    connect(SVM, SIGNAL(compareClicked(QStringList)), this, SLOT(compare(QStringList)));
    ui->stackedWidget->setCurrentIndex(2);
}

void MainWindow::goToRf(QString dataPath)
{
    random_forest *RF = new random_forest(this, dataPath);
    ui->stackedWidget->insertWidget(3, RF);

    connect(RF, SIGNAL(backClicked()), this, SLOT(goBack()));
    connect(RF, SIGNAL(compareClicked(QStringList)), this, SLOT(compare(QStringList)));
    ui->stackedWidget->setCurrentIndex(3);
}

void MainWindow::goBack()
{
    ui->stackedWidget->setCurrentIndex(0);
}

void MainWindow::compare(QStringList classes)
{

    final *finalPage = new final(this, classes);

    ui->stackedWidget->insertWidget(4, finalPage);
    ui->stackedWidget->setCurrentIndex(4);
    connect(finalPage, SIGNAL(homeClicked()), this, SLOT(createNew()));
}

void MainWindow::createNew()
{
    ui->stackedWidget->removeWidget(0);
    start *startPage = new start(this);
    ui->stackedWidget->insertWidget(0, startPage);
    ui->stackedWidget->setCurrentIndex(0);
}





