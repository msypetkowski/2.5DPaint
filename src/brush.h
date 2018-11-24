#pragma once
#include <QtGlobal>
#include <QVector3D>

struct BrushSettings {
	qreal pressure;
	qreal heightPressure;
	qreal size;
	qreal falloff;
	QVector3D color;
};
