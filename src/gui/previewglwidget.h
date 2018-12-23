#ifndef PREVIEWGLWIDGET_H
#define PREVIEWGLWIDGET_H

#include <array>

#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLBuffer>
#include <QOpenGLShaderProgram>
#include <QOpenGLTexture>

#include "../brush_settings.h"
#include "../brush_type.h"
#include "cpu_painter.h"
#include "gpu_painter.h"

class PreviewGLWidget : public QOpenGLWidget, protected QOpenGLFunctions
{
	Q_OBJECT
public:
    PreviewGLWidget(QWidget* parent);
    ~PreviewGLWidget();

	QString getGLinfo();
	void reloadTexture(int, int);

	void setBrushSettings(const BrushSettings& bs) {
		painter()->setBrush(bs);
	}
	void applyBrush(int x, int y);
	void setBrushType(BrushType type);
	void setTexture(QString type, QString file);
	void enableCUDA(bool enable);
	void clearImage();

protected:
    void initializeGL() override;
    void resizeGL(int, int) override;
    void paintGL() override;
	void mousePressEvent(QMouseEvent * event) override;
	void mouseReleaseEvent(QMouseEvent * event) override;
	void mouseMoveEvent(QMouseEvent * event) override;

private:
	QOpenGLBuffer* m_vertices;
	QOpenGLBuffer* m_texcoords;
	QOpenGLBuffer* m_indices;

	QOpenGLBuffer* m_pbo;
	uchar4 *pbo_dptr;

    QOpenGLShaderProgram m_program;
    QOpenGLTexture *m_texture;

	// mouse
	QPoint lastPos;
	int xAtPress, yAtPress;

	int width = -1, height = -1;

	void initShader();
	void initPBO(int, int);
	void deletePBO();
	void initVAO();
	void imageTextureInit(int, int);

	BrushSettings brushSettings;

	bool cuda_enabled;
	std::unique_ptr<GPUPainter> gpu = std::make_unique<GPUPainter>();
	std::unique_ptr<CPUPainter> cpu = std::make_unique<CPUPainter>();

	Painter *painter() {
		return cuda_enabled ? (Painter *)gpu.get() : (Painter *)cpu.get();
	}

public slots:
	void refresh(int, double);

};

#endif // PREVIEWGLWIDGET_H
