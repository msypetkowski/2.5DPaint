#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "../brush_type.h"

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

	assert(connect(ui->defaultBrush, SIGNAL(clicked(bool)), this, SLOT(updateBrushType())));
	assert(connect(ui->texturedBrush, SIGNAL(clicked(bool)), this, SLOT(updateBrushType())));
	assert(connect(ui->radioButton_3, SIGNAL(clicked(bool)), this, SLOT(updateBrushType())));

	updateSettings();
	updateBrushType();
}

MainWindow::~MainWindow()
{
	delete ui;
}

#include <iostream>

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

void MainWindow::updateBrushType() 
{
	const auto brush_id = ui->defaultBrush->isChecked() 
		+ (ui->texturedBrush->isChecked() << 1) 
	    + (ui->radioButton_3->isChecked() << 2);
	ui->openGLWidget->setBrushType(static_cast<BrushType>(brush_id));

}
