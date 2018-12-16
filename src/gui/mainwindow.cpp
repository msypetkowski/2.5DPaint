#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "../brush_type.h"

#include <QFileDialog>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    auto glwidget = ui->openGLWidget;
	assert(connect(ui->enableCudaCheckBox, SIGNAL(clicked(bool)), this, SLOT(enableCuda())));

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

	assert(connect(ui->colorChooseButton, SIGNAL(clicked(bool)), this, SLOT(setColorTexture())));
	assert(connect(ui->heightChooseButton, SIGNAL(clicked(bool)), this, SLOT(setHeightTexture())));

	enableCuda();
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

void MainWindow::updateBrushType()
{
	const auto brush_id = (ui->defaultBrush->isChecked() ? static_cast<int>(BrushType::Default) : 0)
		+ (ui->texturedBrush->isChecked() ? static_cast<int>(BrushType::Textured) : 0)
	    + (ui->radioButton_3->isChecked() ? static_cast<int>(BrushType::Third) : 0);
	ui->openGLWidget->setBrushType(static_cast<BrushType>(brush_id));
}

void MainWindow::enableCuda() {
	ui->openGLWidget->enableCUDA(ui->enableCudaCheckBox->isChecked());
	updateBrushType();
	updateSettings();
}

void MainWindow::setColorTexture()
{
	browseFilesFor(ui->colorFilename);
}

void MainWindow::setHeightTexture()
{
	browseFilesFor(ui->heightFilename);
}

void MainWindow::browseFilesFor(QLabel *what)
{
    const auto file = QFileDialog::getOpenFileName(this, tr("Choose Texture"), QDir::currentPath());
	const auto filename = QFileInfo(file).fileName();
	what->setText("Chosen texture: " + filename);
	ui->openGLWidget->setTexture(what->objectName(), file);
}
