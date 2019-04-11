#include "final.h"
#include "ui_final.h"

final::final(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::final)
{
    ui->setupUi(this);
}

final::~final()
{
    delete ui;
}
