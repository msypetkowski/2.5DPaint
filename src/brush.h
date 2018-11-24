#pragma once
#include <QtGlobal>
#include <QVector3D>

struct BrushSettings {
	qreal pressure;
	qreal heightPressure;
	qreal size;
	QVector3D color;
};
