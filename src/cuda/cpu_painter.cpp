#include "cpu_painter.h"

#include <iostream>

#include <QtMath>
#include <QVector3D>

#include "helper_cuda.h"

qreal normal_from_delta(qreal dx) {
	return dx / qSqrt(dx * dx + 1);
}

qreal lerp(qreal v1, qreal v2, qreal weight) {
	return v1 * (1 - weight) + v2 * weight;
}

Color lerp(Color c1, Color c2, qreal weight) {
	return c1 * (1 - weight) + c2 * weight;
}

qreal cosine_fallof(qreal val, qreal falloff) {
	assert(val >= 0.0);
	assert(val <= 1.0);
	val = qPow(val, falloff);
	return (qCos(val  * M_PI) + 1) * 0.5;
}

Color interpolate_color(Color oldColor, qreal strength, const Color& newColor) {
	Color ret;
	ret = lerp(oldColor, newColor, strength);
	ret = qBound(Color(0,0,0), ret, Color(255,255,255));
	return ret;
}


std::pair<int, int> get_coords(const QImage& image, int x, int y, int w, int h) {
	const auto pixel_x = int(x / float(w) * image.width());
	const auto pixel_y = int(y / float(w) * image.height());
	return { pixel_x, pixel_y };
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
	return bufferHeight[getBufferIndex(x,y)];
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

	Color color = bufferColor[i];
	color = lerp(color * shadow, Color(255, 255, 255), specular);

	// view normals (TODO: remove or make normals visualization feature)
	// color.x = normal.x()*255.0/2 + 255.0/2;
	// color.y = normal.y()*255.0/2 + 255.0/2;
	// color.z = normal.z()*255;

	buffer[i] = color.getAsUchar4();
}

void CPUPainter::updateWholeDisplay() {
	for (int x=0 ; x<w ; ++x) {
		for (int y=0 ; y<h ; ++y) {
			updateDisplay(x,y);
		}
	}
}

void CPUPainter::setDimensions(int w1, int h1, uchar4 *pbo)
{
	w = w1;
	h = h1;

	int buf_size = w * h;

	printf("init/resize cpu buffers (%d, %d)\n", w, h);
	buffer.resize(buf_size);
	bufferColor.resize(buf_size);
	bufferHeight.resize(buf_size);

	checkCudaErrors(cudaMemcpy(&buffer[0], pbo, buf_size * sizeof(uchar4), cudaMemcpyDeviceToHost));

	Color fill(125, 125, 125);
	bufferColor.fill(fill);
	bufferHeight.fill(0.0);
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
			bufferColor[i] = interpolate_color(bufferColor[i], strength, brushSettings.color);

			// paint height
			strength = brushSettings.heightPressure * cosine_fallof(radius / maxRadius, brushSettings.falloff);
			bufferHeight[i] = qBound(-1.0, bufferHeight[i] + strength, 1.0);
		}
	}
	updatePainted(mx, my);
}

void CPUPainter::updatePainted(int mx, int my) {
	qreal maxRadius = brushSettings.size/2;
	for (int x=mx - maxRadius + 1 ; x < mx + maxRadius; ++x) {
		for (int y=my - maxRadius + 1 ; y < my + maxRadius; ++y) {
			if (!inBounds(x,y))
				continue;
			updateDisplay(x,y);
		}
	}
}

void CPUPainter::brushTextured(int mx, int my) {
	qreal maxRadius = brushSettings.size/2;
	if (color_image.isNull() || height_image.isNull()) {
		std::clog << "No texture set\n";
		return;
	}
	for (int x=mx - maxRadius + 1 ; x < mx + maxRadius; ++x) {
		for (int y=my - maxRadius + 1 ; y < my + maxRadius; ++y) {
			if (!inBounds(x,y))
				continue;
			qreal radius = qSqrt((x-mx) * (x-mx) + (y-my) * (y-my));
			if (radius > maxRadius) {
				continue;
			}
			int	i = getBufferIndex(x,y);

			qreal strength = brushSettings.pressure * cosine_fallof(radius / maxRadius, brushSettings.falloff);
			const auto color_coords = 
				get_coords(color_image, x-mx+maxRadius, y-my+maxRadius, maxRadius * 2, maxRadius * 2);
			const auto pixel = color_image.pixel(color_coords.first, color_coords.second);
			bufferColor[i] = interpolate_color(
				bufferColor[i],
				strength,
				Color(qRed(pixel), qGreen(pixel), qBlue(pixel)));

			const auto height_coords = 
				get_coords(height_image, x-mx+maxRadius, y-my+maxRadius, maxRadius * 2, maxRadius * 2);
			const auto height = qRed(height_image.pixel(height_coords.first, height_coords.second)) * 0.001f;
			strength = brushSettings.heightPressure * height * cosine_fallof(radius / maxRadius, brushSettings.falloff);
			bufferHeight[i] = qBound(-1.0, bufferHeight[i] + strength, 1.0);
		}
	}
	updatePainted(mx, my);
}

void CPUPainter::setBrushType(BrushType type) {
	using namespace std::placeholders;
	switch (type) {
		case BrushType::Default:
			paint_function = std::bind(&CPUPainter::brushBasic, this, _1, _2);
			break;
		case BrushType::Textured:
			paint_function = std::bind(&CPUPainter::brushTextured, this, _1, _2);
			break;
		case BrushType::Third:
			std::clog << "Warning: chose unused brush\n";
			break;
		default:
			throw std::runtime_error("Invalid brush type: "
									 + std::to_string(static_cast<int>(type)));
	}
}

void CPUPainter::doPainting(int x, int y, uchar4 *pbo) {
    QElapsedTimer performanceTimer;
	const auto buf_size = w * h * sizeof(uchar4);

	//TODO: leaving it here, this logic used to be in previewGLWidget, although I don't know why
	//performanceTimer.restart();
	//checkCudaErrors(cudaMemcpy(&buffer[0], pbo, buf_size, cudaMemcpyDeviceToHost));
	//const auto memcpy_d2h = performanceTimer.nsecsElapsed();

	performanceTimer.restart();
	paint_function(x, y);
	const auto painting_duration = performanceTimer.nsecsElapsed();

	performanceTimer.restart();
	checkCudaErrors(cudaMemcpy(pbo, &buffer[0], buf_size, cudaMemcpyHostToDevice));
	const auto memcpy_h2d = performanceTimer.nsecsElapsed();

	std::clog << "Painting: " << painting_duration/1e6f << "ms\n";
	std::clog << "Copying: " << memcpy_h2d/1e6f << "ms\n";
}

void CPUPainter::updateBuffer(uchar4 *pbo) {
	if (w < 0 || h < 0)
		return;
	const auto buf_size = w * h * sizeof(uchar4);
	checkCudaErrors(cudaMemcpy(&buffer[0], pbo, buf_size, cudaMemcpyDeviceToHost));
}

void CPUPainter::setTexture(const QString& type, const QString& path) {
	if (type == "colorFilename") {
		color_image = QImage(path);
	} else {
		height_image = QImage(path);
	}
}

std::pair<const QImage&, const QImage&> CPUPainter::getTextures(const QString& type) const {
	if (type == "colorFilename") {
		return color_image;
	} else {
		return height_image;
	}
}

