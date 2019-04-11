#include "start.h"
#include "ui_start.h"

#include <PythonQt.h>
#include <QDebug>
#include <QDirIterator>

#include <QFileDialog>
#include <QLineEdit>
#include <QPixmap>

start::start(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::start)
{
    ui->setupUi(this);

    ui->nextButton_start->setEnabled(false);

    ui->comboBox->addItem("Convolutional Neural Networks");
    ui->comboBox->addItem("Support Vector Machines");
    ui->comboBox->addItem("Random Forest");

    ui->dirName->setVisible(false);
    ui->dirName_2->setVisible(false);
    ui->dirName_3->setVisible(false);
    ui->dirName_4->setVisible(false);
    ui->dirName_5->setVisible(false);
    ui->dirName_6->setVisible(false);

    ui->imgLabel->setVisible(false);
    ui->imgLabel_2->setVisible(false);
    ui->imgLabel_3->setVisible(false);
    ui->imgLabel_4->setVisible(false);
    ui->imgLabel_5->setVisible(false);
    ui->imgLabel_6->setVisible(false);

    connect(this, SIGNAL(datasetSelected()), this, SLOT(showImages()));
    connect(this, SIGNAL(datasetSelected()), this, SLOT(showNextButton()));


}

start::~start()
{
    delete ui;
}

void start::on_browseButton_clicked()
{
    QString dir = QFileDialog::getExistingDirectory(this, tr("Open Directory"), "/home", QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);
    ui->datapath_lineEdit->setText(dir);

    QDir rootDir(dir);
    if(rootDir.entryList(QDir::Dirs | QDir::NoDotAndDotDot).size() != 3)
    {
        setImages(rootDir);
    }
    else
    {
        QDir cnnDir(dir + "/train");
        setImages(cnnDir);
    }
    emit datasetSelected();
}

void start::setImages(QDir dirDir)
{
    QStringList dirList = dirDir.entryList(QDir::Dirs | QDir::NoDotAndDotDot);

    ui->dirName->setText(dirList.at(0));
    ui->dirName_2->setText(dirList.at(1));
    ui->dirName_3->setText(dirList.at(2));
    ui->dirName_4->setText(dirList.at(3));
    ui->dirName_5->setText(dirList.at(4));
    ui->dirName_6->setText(dirList.at(5));

    QStringList imgPath;

    for(int i = 0; i < dirList.length(); i++)
    {
        QDir dirPath(dirDir.filePath(dirList.at(i)));
        QStringList imgList = dirPath.entryList(QDir::Files | QDir::NoDotAndDotDot);

        // nr = rand() % 100;
        imgPath.append(dirPath.filePath(imgList.at(3)));
    }

    QPixmap image1(imgPath.at(0));
    image1 = image1.scaled(QSize(128, 128),Qt::KeepAspectRatio, Qt::SmoothTransformation);
    ui->imgLabel->setPixmap(image1);

    QPixmap image2(imgPath.at(1));
    image2 = image2.scaled(QSize(128, 128),Qt::KeepAspectRatio, Qt::SmoothTransformation);
    ui->imgLabel_2->setPixmap(image2);

    QPixmap image3(imgPath.at(2));
    image3 = image3.scaled(QSize(128, 128),Qt::KeepAspectRatio, Qt::SmoothTransformation);
    ui->imgLabel_3->setPixmap(image3);

    QPixmap image4(imgPath.at(3));
    image4 = image4.scaled(QSize(128, 128),Qt::KeepAspectRatio, Qt::SmoothTransformation);
    ui->imgLabel_4->setPixmap(image4);

    QPixmap image5(imgPath.at(4));
    image5 = image5.scaled(QSize(128, 128),Qt::KeepAspectRatio, Qt::SmoothTransformation);
    ui->imgLabel_5->setPixmap(image5);

    QPixmap image6(imgPath.at(5));
    image6 = image6.scaled(QSize(128, 128),Qt::KeepAspectRatio, Qt::SmoothTransformation);
    ui->imgLabel_6->setPixmap(image6);
}

void start::on_nextButton_start_clicked()
{
    if(ui->comboBox->currentText() == "Convolutional Neural Networks")
    {
        emit cnnNextClicked(ui->datapath_lineEdit->text());
    }
    else if(ui->comboBox->currentText() == "Support Vector Machines")
    {
        emit svmNextClicked(ui->datapath_lineEdit->text());
    }
    else
    {
        emit rfNextClicked(ui->datapath_lineEdit->text());
    }
}

void start::showImages()
{
    ui->dirName->setVisible(true);
    ui->dirName_2->setVisible(true);
    ui->dirName_3->setVisible(true);
    ui->dirName_4->setVisible(true);
    ui->dirName_5->setVisible(true);
    ui->dirName_6->setVisible(true);

    ui->imgLabel->setVisible(true);
    ui->imgLabel_2->setVisible(true);
    ui->imgLabel_3->setVisible(true);
    ui->imgLabel_4->setVisible(true);
    ui->imgLabel_5->setVisible(true);
    ui->imgLabel_6->setVisible(true);
}

void start::showNextButton()
{
    ui->nextButton_start->setEnabled(true);
}
