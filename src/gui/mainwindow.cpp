#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    auto glwidget = ui->openGLWidget;
    auto checkbox = ui->enableCudaCheckBox;
    assert(connect(checkbox, SIGNAL(clicked(bool)), glwidget, SLOT(enableCUDA(bool))));
}

MainWindow::~MainWindow()
{
    delete ui;
}
