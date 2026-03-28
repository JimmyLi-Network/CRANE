CC = xcrun clang
CFLAGS = -O2 -Wall -Wno-deprecated-declarations -fobjc-arc -fPIC
FRAMEWORKS = -framework Foundation -framework IOSurface -ldl
BRIDGE_DIR = src
TARGET = $(BRIDGE_DIR)/libane_bridge.dylib

all: $(TARGET)

$(TARGET): $(BRIDGE_DIR)/ane_bridge.m $(BRIDGE_DIR)/ane_bridge.h
	$(CC) $(CFLAGS) -dynamiclib -o $@ $(BRIDGE_DIR)/ane_bridge.m $(FRAMEWORKS)

clean:
	rm -f $(TARGET)

test: $(TARGET)
	ARC_ANE_BRIDGE_PATH=$(TARGET) python -m pytest tests/ -q

.PHONY: all clean test
