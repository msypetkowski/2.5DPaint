#include "kernel_cpu.h"
#include <QtMath>
#include <QVector3D>

extern QVector<uchar4> cpu_buffer=QVector<uchar4>();
extern QVector<uchar4> cpu_buffer_color=QVector<uchar4>();
extern QVector<qreal> cpu_buffer_height=QVector<qreal>();

// TODO: make class not globals
static int w, h;

int get_buffer_index(int x, int y) {
	return w - 1 - x + (y * w);
}

bool in_bounds(int x, int y) {
	return x >= 0 && x < w && y >= 0 && y < h;
}

qreal sample_height(int x, int y) {
	x = qBound(0, x, w-1);
	y = qBound(0, y, h-1);
	return cpu_buffer_height[get_buffer_index(x,y)];
}

qreal normal_from_delta(qreal dx) {
	return dx / qSqrt(dx * dx + 1);
}

QVector3D get_normal(int x, int y) {
	qreal dx=0, dy=0;

	auto mid = sample_height(x,y);
	auto left = sample_height(x-1, y);
	auto right = sample_height(x+1, y);
	auto top = sample_height(x, y+1);
	auto bottom = sample_height(x, y-1);

	dx += normal_from_delta(mid - right) / 2;
	dx -= normal_from_delta(mid - left) / 2;

	dy += normal_from_delta(mid - top) / 2;
	dy -= normal_from_delta(mid - bottom) / 2;

	// TODO: make parameter or constant
	dx *=200;
	dy *=200;

	dx = dx / sqrt(dx*dx + dy*dy + 1);
	dy = dy / sqrt(dx*dx + dy*dy + 1);

	assert(qAbs(dx) <= 1);
	assert(qAbs(dy) <= 1);
	auto ret = QVector3D(dx, dy, sqrt(1 - dx*dx - dy*dy));
	ret.normalize();
	return ret;
}

qreal lerp(qreal v1, qreal v2, qreal weight) {
	return v1 * (1 - weight) + v2 * weight;
}

void update_display(int x, int y) {
	int	i = get_buffer_index(x,y);

	auto normal = get_normal(x,y);

	QVector3D lighting(0.07, 0.07, 1.0);
	lighting.normalize();

	// TODO: use lighting vector here
	qreal shadow = normal.z() * 0.80 - normal.x()*0.1 - normal.y()*0.1 + (sample_height(x,y)) / 4;
	shadow = qBound(0.0, shadow, 1.0);


	qreal specular = 1 - (normal - lighting).length();
	specular = qPow(specular, 8);
	specular = qBound(0.0, specular, 1.0);

	uchar4 color = cpu_buffer_color[i];
	color.x = lerp(color.x * shadow, 255, specular);
	color.y = lerp(color.y * shadow, 255, specular);
	color.z = lerp(color.z * shadow, 255, specular);

	// view normals (TODO: remove or make normals visualization feature)
	// color.x = normal.x()*255.0/2 + 255.0/2;
	// color.y = normal.y()*255.0/2 + 255.0/2;
	// color.z = normal.z()*255;

	cpu_buffer[i] = color;
}

void update_whole_display(int w1, int h1) {
	w=w1;
	h=h1;
	for (int x=0 ; x<w ; ++x) {
		for (int y=0 ; y<h ; ++y) {
			update_display(x,y);
		}
	}
}

qreal cosine_fallof(qreal val, qreal falloff) {
	assert(val >= 0.0);
	assert(val <= 1.0);
	val = qPow(val, falloff);
	return (qCos(val  * M_PI) + 1) * 0.5;
}

uchar4 interpolate_color(uchar4 inputColor, qreal strength, const BrushSettings& bs) {
	uchar4 ret;
	ret.x = qBound(0.0, lerp(inputColor.x, bs.color.x() * 255.0, strength), 255.0);
	ret.y = qBound(0.0, lerp(inputColor.y, bs.color.y() * 255.0, strength), 255.0);
	ret.z = qBound(0.0, lerp(inputColor.z, bs.color.z() * 255.0, strength), 255.0);
	return ret;
}

void brush_basic(int w1, int h1, int mx, int my, const BrushSettings& bs) {
	w=w1;
	h=h1;
	qreal maxRadius = bs.size/2;
	for (int x=mx - maxRadius + 1 ; x < mx + maxRadius; ++x) {
		for (int y=my - maxRadius + 1 ; y < my + maxRadius; ++y) {
			if (!in_bounds(x,y))
				continue;

			qreal radius = qSqrt((x-mx) * (x-mx) + (y-my) * (y-my));
			if (radius > maxRadius) {
				continue;
			}
			int	i = get_buffer_index(x,y);

			// paint color
			qreal strength = bs.pressure * cosine_fallof(radius / maxRadius, bs.falloff);
			cpu_buffer_color[i] = interpolate_color(cpu_buffer_color[i], strength, bs);

			// paint height
			strength = bs.heightPressure * cosine_fallof(radius / maxRadius, bs.falloff);
			cpu_buffer_height[i] = qBound(-1.0, cpu_buffer_height[i] + strength, 1.0);
		}
	}

	for (int x=mx - maxRadius + 1 ; x < mx + maxRadius; ++x) {
		for (int y=my - maxRadius + 1 ; y < my + maxRadius; ++y) {
			if (!in_bounds(x,y))
				continue;
			update_display(x,y);
		}
	}
}
