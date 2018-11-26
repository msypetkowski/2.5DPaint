#include "kernel_cpu.h"
#include <QtMath>
#include <QVector3D>

// TODO: make class not globals


qreal normal_from_delta(qreal dx) {
	return dx / qSqrt(dx * dx + 1);
}

qreal lerp(qreal v1, qreal v2, qreal weight) {
	return v1 * (1 - weight) + v2 * weight;
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

int CPUPainter::getBufferIndex(int x, int y) {
	return w - 1 - x + (y * w);
}

bool CPUPainter::inBounds(int x, int y) {
	return x >= 0 && x < w && y >= 0 && y < h;
}


qreal CPUPainter::sampleHeight(int x, int y) {
	x = qBound(0, x, w-1);
	y = qBound(0, y, h-1);
	return cpuBufferHeight[getBufferIndex(x,y)];
}


QVector3D CPUPainter::getNormal(int x, int y) {
	qreal dx=0, dy=0;

	auto mid = sampleHeight(x,y);
	auto left = sampleHeight(x-1, y);
	auto right = sampleHeight(x+1, y);
	auto top = sampleHeight(x, y+1);
	auto bottom = sampleHeight(x, y-1);

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

void CPUPainter::updateDisplay(int x, int y) {
	int	i = getBufferIndex(x,y);

	auto normal = getNormal(x,y);

	QVector3D lighting(0.07, 0.07, 1.0);
	lighting.normalize();

	// TODO: use lighting vector here
	qreal shadow = normal.z() * 0.80 - normal.x()*0.1 - normal.y()*0.1 + (sampleHeight(x,y)) / 4;
	shadow = qBound(0.0, shadow, 1.0);


	qreal specular = 1 - (normal - lighting).length();
	specular = qPow(specular, 8);
	specular = qBound(0.0, specular, 1.0);

	uchar4 color = cpuBufferColor[i];
	color.x = lerp(color.x * shadow, 255, specular);
	color.y = lerp(color.y * shadow, 255, specular);
	color.z = lerp(color.z * shadow, 255, specular);

	// view normals (TODO: remove or make normals visualization feature)
	// color.x = normal.x()*255.0/2 + 255.0/2;
	// color.y = normal.y()*255.0/2 + 255.0/2;
	// color.z = normal.z()*255;

	cpuBuffer[i] = color;
}

void CPUPainter::updateWholeDisplay() {
	for (int x=0 ; x<w ; ++x) {
		for (int y=0 ; y<h ; ++y) {
			updateDisplay(x,y);
		}
	}
}

void CPUPainter::setDimensions(int w1, int h1)
{
	w = w1;
	h = h1;

	int buf_size = w * h;

	printf("init/resize cpu buffers (%d, %d)\n", w, h);
	cpuBuffer.resize(buf_size);
	cpuBufferColor.resize(buf_size);
	cpuBufferHeight.resize(buf_size);

	uchar4 fill;
	fill.x = 125; fill.y = 125; fill.z = 125; fill.w = 255;
	cpuBufferColor.fill(fill);
	cpuBufferHeight.fill(0.0);
	// cpu_buffer.fill(fill);
	updateWholeDisplay();
}

void CPUPainter::brushBasic(int mx, int my) {
	qreal maxRadius = brushSettings.size/2;
	for (int x=mx - maxRadius + 1 ; x < mx + maxRadius; ++x) {
		for (int y=my - maxRadius + 1 ; y < my + maxRadius; ++y) {
			if (!inBounds(x,y))
				continue;

			qreal radius = qSqrt((x-mx) * (x-mx) + (y-my) * (y-my));
			if (radius > maxRadius) {
				continue;
			}
			int	i = getBufferIndex(x,y);

			// paint color
			qreal strength = brushSettings.pressure * cosine_fallof(radius / maxRadius, brushSettings.falloff);
			cpuBufferColor[i] = interpolate_color(cpuBufferColor[i], strength, brushSettings);

			// paint height
			strength = brushSettings.heightPressure * cosine_fallof(radius / maxRadius, brushSettings.falloff);
			cpuBufferHeight[i] = qBound(-1.0, cpuBufferHeight[i] + strength, 1.0);
		}
	}

	for (int x=mx - maxRadius + 1 ; x < mx + maxRadius; ++x) {
		for (int y=my - maxRadius + 1 ; y < my + maxRadius; ++y) {
			if (!inBounds(x,y))
				continue;
			updateDisplay(x,y);
		}
	}
}
