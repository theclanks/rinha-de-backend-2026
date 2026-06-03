CC = gcc
CFLAGS = -O3 -mavx -mavx2 -mfma -shared -fPIC -I$(ERTS_INCLUDE_DIR)
LDFLAGS = -lm

ERTS_INCLUDE_DIR ?= $(shell erl -noshell -eval 'io:format("~s", [code:lib_dir(erts) ++ "/include"]), halt().' 2>/dev/null)

all: priv/knn.so

priv/knn.so: knn.c
	mkdir -p priv
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -rf priv

.PHONY: all clean
