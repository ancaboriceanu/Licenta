#ifndef FINAL_H
#define FINAL_H

#include <QWidget>

namespace Ui {
class final;
}

class final : public QWidget
{
    Q_OBJECT

public:
    explicit final(QWidget *parent = 0);
    ~final();

private:
    Ui::final *ui;
};

#endif // FINAL_H
