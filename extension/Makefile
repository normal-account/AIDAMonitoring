EXTENSION = aida_hook
EXTVERSION = 1.0

MODULE_big = aida_hook
OBJS = aida_hook.o

DATA = aida_hook--1.0.sql

PYTHON_INCLUDES := $(shell python3.10-config --includes)
PG_CPPFLAGS += $(PYTHON_INCLUDES)

PYTHON_LDFLAGS := $(shell python3.10-config --ldflags)
SHLIB_LINK += $(PYTHON_LDFLAGS)
SHLIB_LINK += -Wl,-E
SHLIB_LINK += -lpython3.10

PG_CONFIG ?= pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)

include $(PGXS)