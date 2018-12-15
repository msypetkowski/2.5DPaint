#ifndef PREVIEWGLWIDGET_H
#define PREVIEWGLWIDGET_H

#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLBuffer>
#include <QOpenGLShaderProgram>
#include <QOpenGLTexture>
#include <QtCore/QElapsedTimer>

#include "../brush_settings.h"
#include "../brush_type.h"
#include "cpu_painter.h"

class PreviewGLWidget : public QOpenGLWidget, protected QOpenGLFunctions
{
	Q_OBJECT
public:
    PreviewGLWidget(QWidget* parent);
    ~PreviewGLWidget();

	QString getGLinfo();
	void reloadTexture(int, int);

	void setBrushSettings(const BrushSettings& bs) {
		painter->setBrush(bs);
	}
	void setBrushType(BrushType type);
	void setTexture(QString type, QString file);

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

    QOpenGLShaderProgram m_program;
    QOpenGLTexture *m_texture;

	// mouse
	QPoint lastPos;
	int xAtPress, yAtPress;

	int width, height;

	void initShader();
	void initPBO(int, int);
	void deletePBO();
	void initVAO();
	void imageTextureInit(int, int);

	BrushSettings brushSettings;

    QElapsedTimer performanceTimer;

	std::unique_ptr<Painter> painter;

public slots:
	void refresh(int, double);

	void enableCUDA(bool enable);
};

#endif // PREVIEWGLWIDGET_H
