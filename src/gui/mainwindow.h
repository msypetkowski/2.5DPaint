#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

#include "../brush_settings.h"

namespace Ui {
class MainWindow;
}

class QLabel;

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

public slots:
	void updateSettings();
	void updateBrushType();
	void setHeightTexture();
	void setColorTexture();

private:
	void browseFilesFor(QLabel *what);

    Ui::MainWindow *ui;

	BrushSettings brushSettings;
};

#endif // MAINWINDOW_H
