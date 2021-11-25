TARGET := gemm.cubin

all: $(TARGET)

$(TARGET): gemm.cu
	nvcc -arch sm_75 -maxrregcount 128 --ptxas-options=-v -cubin gemm.cu

clean:
	rm -f $(TARGET)

.PHONY: all clean 
