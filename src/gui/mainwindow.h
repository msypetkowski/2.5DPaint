#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

#include "../brush.h"

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

public slots:
	void updateSettings();
	void updateBrushType();

private:
    Ui::MainWindow *ui;

	BrushSettings brushSettings;
};

#endif // MAINWINDOW_H
