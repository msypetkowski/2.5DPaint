#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    auto glwidget = ui->openGLWidget;
	assert(connect(ui->enableCudaCheckBox, SIGNAL(clicked(bool)), glwidget, SLOT(enableCUDA(bool))));
	glwidget->enableCUDA(ui->enableCudaCheckBox->isChecked());

	assert(connect(ui->brushPressureSpinBox, SIGNAL(valueChanged(qreal)), this, SLOT(updateSettings())));
	assert(connect(ui->heightPressureSpinBox, SIGNAL(valueChanged(qreal)), this, SLOT(updateSettings())));
	assert(connect(ui->brushSizeSpinBox, SIGNAL(valueChanged(qreal)), this, SLOT(updateSettings())));

	assert(connect(ui->brushSizeSpinBox, SIGNAL(valueChanged(qreal)), this, SLOT(updateSettings())));
	assert(connect(ui->brushFalloff, SIGNAL(valueChanged(qreal)), this, SLOT(updateSettings())));

	assert(connect(ui->colorR, SIGNAL(valueChanged(qreal)), this, SLOT(updateSettings())));
	assert(connect(ui->colorG, SIGNAL(valueChanged(qreal)), this, SLOT(updateSettings())));
	assert(connect(ui->colorB, SIGNAL(valueChanged(qreal)), this, SLOT(updateSettings())));

	updateSettings();
}

MainWindow::~MainWindow()
{
	delete ui;
}

void MainWindow::updateSettings()
{
	brushSettings.pressure = ui->brushPressureSpinBox->value();
	brushSettings.heightPressure = ui->heightPressureSpinBox->value();
	brushSettings.size = ui->brushSizeSpinBox->value();
	brushSettings.falloff = ui->brushFalloff->value();

	brushSettings.color.x = ui->colorR->value() * 255;
	brushSettings.color.y = ui->colorG->value() * 255;
	brushSettings.color.z = ui->colorB->value() * 255;

	ui->openGLWidget->setBrushSettings(brushSettings);
}
