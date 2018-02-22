

all:

OPENFST_CXXFLAGS = 
OPENFST_LDLIBS =


include ../kaldi.mk

LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)

TESTFILES = cu-vector-test cu-matrix-test cu-math-test cu-test cu-sp-matrix-test cu-packed-matrix-test cu-tp-matrix-test \
            cu-block-matrix-test cu-matrix-speed-test cu-vector-speed-test cu-sp-matrix-speed-test cu-array-test


OBJFILES = cu-device.o cu-math.o cu-matrix.o cu-packed-matrix.o cu-sp-matrix.o \
           cu-vector.o cu-common.o cu-tp-matrix.o cu-rand.o cu-block-matrix.o
ifeq ($(CUDA), true)
  OBJFILES += cu-kernels.o cu-randkernels.o
endif

LIBNAME = kaldi-cudamatrix

all:  $(LIBFILE)


ifeq ($(CUDA), true)
  #Default compute capability architectures we compile with
  CUDA_ARCH=-gencode arch=compute_61,code=sm_61 \
	    -gencode arch=compute_60,code=sm_60
endif


#implicit rule for kernel compilation
%.o : %.cu
	$(CUDATKDIR)/bin/nvcc -c $< -o $@ $(CUDA_INCLUDE) $(CUDA_FLAGS) $(CUDA_ARCH) -I../


ADDLIBS = ../matrix/kaldi-matrix.a ../base/kaldi-base.a  ../util/kaldi-util.a 

include ../makefiles/default_rules.mk

