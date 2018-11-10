#include "mainwindow.h"
#include <QApplication>
#include "kernel.h"

int main(int argc, char *argv[])
{
	setupCuda();

    QApplication a(argc, argv);
    MainWindow w;
    w.show();

    return a.exec();
}
