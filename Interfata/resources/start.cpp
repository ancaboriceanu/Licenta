#include "start.h"
#include "ui_start.h"

#include <PythonQt.h>
#include <QDebug>
#include <QDirIterator>

#include <QFileDialog>
#include <QLineEdit>
#include <QPixmap>

#include <QFileSystemWatcher>

start::start(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::start)
{
    ui->setupUi(this);

    ui->nextButton_start->setEnabled(false);

    ui->comboBox->addItem("Convolutional Neural Networks");
    ui->comboBox->addItem("Support Vector Machines");
    ui->comboBox->addItem("Random Forest");

    connect(this, SIGNAL(datasetSelected()), this, SLOT(showNextButton()));
}

start::~start()
{
    delete ui;
}


void start::on_browseButton_clicked()
{
    QString datasetDirectory = QFileDialog::getExistingDirectory(this, tr("Open Directory"), "/home", QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);
    ui->datapath_lineEdit->setText(datasetDirectory);

    QDir completeDatasetDir(datasetDirectory);
    if(completeDatasetDir.entryList(QDir::Dirs | QDir::NoDotAndDotDot).size() != 3)
    {
        setImages(completeDatasetDir);
    }
    else
    {
        QDir cnnDir(datasetDirectory + "/train");
        setImages(cnnDir);
    }
    emit datasetSelected();
}

void start::setImages(QDir datasetPath)
{
    QStringList directoriesList = datasetPath.entryList(QDir::Dirs | QDir::NoDotAndDotDot);
    QStringList imagesList;

    QLayoutItem *labelToRemove;
    QLayoutItem *imageToRemove;

    while (((labelToRemove = ui->labelsHorizontalLayout->takeAt(0)) != 0) && ((imageToRemove = ui->imagesHorizontalLayout->takeAt(0)) != 0))
    {
        delete labelToRemove;
        delete imageToRemove;
    }

    for (int i = 0; i < directoriesList.length(); i++)
    {
        QLabel *label = new QLabel();
        label->setAlignment(Qt::AlignCenter);
        QFont f( "Seagoe UI", 10);
        label->setFont(f);
        label->setText((directoriesList.at(i)));
        label->setMinimumHeight(25);
        ui->labelsHorizontalLayout->addWidget(label);
    }

    for(int i = 0; i < directoriesList.length(); i++)
    {
        QDir directoryPath(datasetPath.filePath(directoriesList.at(i)));
        QStringList directoryImagesList = directoryPath.entryList(QDir::Files | QDir::NoDotAndDotDot);
        int imageNumber = rand() / RAND_MAX * directoryImagesList.length();
        imagesList.append(directoryPath.filePath(directoryImagesList.at(imageNumber)));
    }

    for(int i = 0; i < imagesList.length(); i++)
    {
        QPixmap image(imagesList.at(i));
        image = image.scaled(QSize(128, 128),Qt::KeepAspectRatio, Qt::SmoothTransformation);
        QLabel *label = new QLabel();
        label->setPixmap(image);
        label->setAlignment(Qt::AlignCenter);
        label->setMinimumHeight(140);
        ui->imagesHorizontalLayout->addWidget(label);
    }
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

void start::showNextButton()
{
    ui->nextButton_start->setEnabled(true);
}

