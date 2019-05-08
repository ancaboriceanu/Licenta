#ifndef START_H
#define START_H

#include <QWidget>
#include <QLabel>
#include <QDir>

namespace Ui {
class start;
}

class start : public QWidget
{
    Q_OBJECT

public:
    explicit start(QWidget *parent = 0);
    ~start();

private slots:
    void on_browseButton_clicked();
    void on_nextButton_start_clicked();
    void showNextButton();

signals:
    void cnnNextClicked(QString dataPath);
    void svmNextClicked(QString dataPath);
    void rfNextClicked(QString dataPath);
    void datasetSelected();

private:
    void setImages(QDir datasetPath);
    void initialization();

private:
    Ui::start *ui;
};

#endif // START_H
