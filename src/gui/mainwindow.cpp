#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "../brush_type.h"

#include <QFileDialog>
#include <QtWidgets/QColorDialog>

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

    assert(connect(ui->colorR, SIGNAL(valueChanged(qreal)), this, SLOT(updateColorBtnBackground())));
    assert(connect(ui->colorG, SIGNAL(valueChanged(qreal)), this, SLOT(updateColorBtnBackground())));
    assert(connect(ui->colorB, SIGNAL(valueChanged(qreal)), this, SLOT(updateColorBtnBackground())));

    assert(connect(ui->defaultBrush, SIGNAL(clicked(bool)), this, SLOT(updateBrushType())));
    assert(connect(ui->texturedBrush, SIGNAL(clicked(bool)), this, SLOT(updateBrushType())));
    assert(connect(ui->smoothBrush, SIGNAL(clicked(bool)), this, SLOT(updateBrushType())));
    assert(connect(ui->inflateBrush, SIGNAL(clicked(bool)), this, SLOT(updateBrushType())));

    assert(connect(ui->colorChooseButton, SIGNAL(clicked(bool)), this, SLOT(setColorTexture())));
    assert(connect(ui->heightChooseButton, SIGNAL(clicked(bool)), this, SLOT(setHeightTexture())));

    assert(connect(ui->colorBtn, SIGNAL(clicked(bool)), this, SLOT(setBrushColor())));
    assert(connect(ui->actionClear, SIGNAL(triggered()), this, SLOT(clearImage())));
    assert(connect(ui->clearBtn, SIGNAL(clicked(bool)), this, SLOT(clearImage())));

    assert(connect(ui->normalCheckbox, SIGNAL(clicked(bool)), this, SLOT(updateSettings())));
    assert(connect(ui->normalBend, SIGNAL(valueChanged(qreal)), this, SLOT(updateSettings())));

    enableCuda();
    updateSettings();
    updateBrushType();
    updateColorBtnBackground();
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

    brushSettings.renderNormals = ui->normalCheckbox->isChecked();
    brushSettings.normalBending = ui->normalBend->value();

    ui->openGLWidget->setBrushSettings(brushSettings);
}

void MainWindow::updateBrushType()
{
    const auto brush_id = (ui->defaultBrush->isChecked() ? static_cast<int>(BrushType::Default) : 0)
        + (ui->texturedBrush->isChecked() ? static_cast<int>(BrushType::Textured) : 0)
        + (ui->smoothBrush->isChecked() ? static_cast<int>(BrushType::Smooth) : 0)
        + (ui->inflateBrush->isChecked() ? static_cast<int>(BrushType::Inflate) : 0);
    ui->openGLWidget->setBrushType(static_cast<BrushType>(brush_id));

    bool isTexturedBrush = brush_id == static_cast<const int>(BrushType::Textured);
    bool isBasicBrush = brush_id == static_cast<const int>(BrushType::Default);
    bool isSmoothBrush = brush_id == static_cast<const int>(BrushType::Smooth);
    bool isInflateBrush = brush_id == static_cast<const int>(BrushType::Inflate);

    ui->colorBtn->setDisabled(!isBasicBrush);
    ui->colorR->setDisabled(!isBasicBrush);
    ui->colorG->setDisabled(!isBasicBrush);
    ui->colorB->setDisabled(!isBasicBrush);
    ui->brushPressureSpinBox->setDisabled(isSmoothBrush);
    ui->heightPressureSpinBox->setDisabled(isSmoothBrush);

    // only for textures
    ui->colorChooseButton->setDisabled(!isTexturedBrush);
    ui->heightChooseButton->setDisabled(!isTexturedBrush);
    ui->colorFilename->setDisabled(!isTexturedBrush);
    ui->heightFilename->setDisabled(!isTexturedBrush);
    ui->texturesLabel->setDisabled(!isTexturedBrush);
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
    if (!file.isNull()) {
        const auto filename = QFileInfo(file).fileName();
        what->setText(filename);
        ui->openGLWidget->setTexture(what->objectName(), file);
    }
}

void MainWindow::setBrushColor()
{
    QColor currentColor = QColor(ui->colorR->value() * 255, ui->colorG->value() * 255, ui->colorB->value() * 255);
    QColor color = QColorDialog::getColor(currentColor, this, "Pick brush color");

    ui->colorR->setValue(color.redF());
    ui->colorG->setValue(color.greenF());
    ui->colorB->setValue(color.blueF());
}

void MainWindow::updateColorBtnBackground()
{
    ui->colorBtn->setStyleSheet(QString("background-color: rgb(%1, %2, %3);").arg(QString::number(255 * ui->colorR->value()),
                                                                                  QString::number(255 * ui->colorG->value()),
                                                                                  QString::number(255 * ui->colorB->value())));
}

void MainWindow::clearImage()
{
    ui->openGLWidget->clearImage();
}
