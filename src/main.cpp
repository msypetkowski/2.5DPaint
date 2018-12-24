#include "mainwindow.h"
#include <QApplication>

#include <string>
#include "tests.h"

int main(int argc, char *argv[])
{
	if (argc > 1 && std::string(argv[1]) == std::string("--test")) {
		return runTests();
	} else  {
		QApplication a(argc, argv);
		MainWindow w;
		w.show();

		return a.exec();
	}
}
