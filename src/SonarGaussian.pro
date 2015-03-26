TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cpp \
    Sonar.cpp \
    Gaussian.cpp

#LIBS += -L/usr/lib -lCGAL -lboost_system -lboost_thread -lprotobuf

LIBS += -I/usr/local/include/opencv -I/usr/local/include/opencv2 -L/usr/local/lib/ -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_contrib -lopencv_legacy -I /usr/local/lib/

HEADERS += \
    Sonar.h \
    Gaussian.h
