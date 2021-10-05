#
# Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
#
NCCL_HOME:=../../build
CUDA_HOME:=/usr/local/cuda
INC:= -I$(NCCL_HOME)/include -I$(CUDA_HOME)/include
# PLUGIN_SO:=libnccl-net.dylib
PLUGIN_SO:=libnccl-net.so

default: $(PLUGIN_SO)

$(PLUGIN_SO): plugin.c
# $(CC) $(INC) -fPIC -shared -o $@ -Wl,-soname,$(PLUGIN_SO) $^
	$(CC) $(INC) -fPIC -shared -o $@ $^
	mkdir -p ${BUILDDIR}lib
	cp $@ ${NCCL_HOME}/lib

clean:
	rm -f ${NCCL_HOME}/lib/$(PLUGIN_SO)
	rm -f $(PLUGIN_SO)
