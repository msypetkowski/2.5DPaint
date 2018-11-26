#pragma once
#include <QtGlobal>
#include <QVector3D>

#include "cuda_runtime.h"

class Color {
public:
	uchar4 getAsUchar4() {
		uchar4 ret;
		ret.x = x;
		ret.y = y;
		ret.z = z;
		ret.w = 255;
		return ret;
	}

	Color() {}
	Color(int x, int y, int z): x(x), y(y), z(z) {}
	Color operator+(qreal v)const { return Color(x+v, y+v, z+v); }
	Color operator*(qreal v)const { return Color(x*v, y*v, z*v); }
	Color operator+(const Color& c)const { return Color(x+c.x, y+c.y, z+c.z); }
	Color operator*(const Color& c)const { return Color(x*c.x, y*c.y, z*c.z); }

	bool operator<(const Color& c)const { return std::tuple<int, int, int>(x,y,z) < std::tuple<int, int, int>(c.x,c.y,c.z); }

	int x, y, z;
};


struct BrushSettings {
	qreal pressure;
	qreal heightPressure;
	qreal size;
	qreal falloff;
	Color color;
};
