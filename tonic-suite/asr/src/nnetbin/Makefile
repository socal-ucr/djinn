COMMON=../../../../common/

include $(COMMON)/Makefile.config

all:
EXTRA_CXXFLAGS = -Wno-sign-compare \
								 -O3 \
								 -fpermissive \
								 -I$(COMMON)/include \
								 -I/home/wong936/djinn/caffe-1.0/$(CAFFE)/include \
								 -I$(CUDA)/include \
								 -I/usr/include/hdf5/serial/ 

ifeq ($(CPU_ONLY), 1)
	EXTRA_CXXFLAGS   += -DCPU_ONLY
endif

include ../kaldi.mk

LDFLAGS += $(CUDA_LDFLAGS) -L/usr/lib/x86_64-linux-gnu/hdf5/serial/
LDLIBS += $(CUDA_LDLIBS) $(COMMON)/libtonic.a \
					/home/wong936/djinn/caffe/$(CAFFE)/lib/libcaffe.so \
					-lglog -lboost_system


BINFILES = nnet-train-frmshuff \
        nnet-train-perutt \
        nnet-train-mmi-sequential \
        nnet-train-mpe-sequential \
        rbm-train-cd1-frmshuff rbm-convert-to-nnet \
        nnet-forward nnet-copy nnet-info nnet-concat \
        transf-to-nnet cmvn-to-nnet nnet-initialize \
        nnet-kl-hmm-acc nnet-kl-hmm-mat-to-component 

TESTFILES = 

ADDLIBS = ../nnet/kaldi-nnet.a ../cudamatrix/kaldi-cudamatrix.a ../lat/kaldi-lat.a \
          ../hmm/kaldi-hmm.a ../tree/kaldi-tree.a ../matrix/kaldi-matrix.a \
          ../util/kaldi-util.a ../base/kaldi-base.a

include ../makefiles/default_rules.mk
