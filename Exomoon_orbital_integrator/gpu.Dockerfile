FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    ML_DEVICE=cuda \
    HNN_MODEL_DIR=/app/src/models_hnn_hill_hinge4

WORKDIR /app

COPY requirements_gpu.txt .
RUN pip install --no-cache-dir -r requirements_gpu.txt

# Numba CUDA requires bare .so names (libcudart.so, libnvrtc.so, libnvvm.so) but
# pip-installed nvidia packages only ship versioned names (.so.12, .so.4 etc).
# Create the missing unversioned symlinks so numba's ctypes loader can find them.
# Also expose all nvidia lib directories via LD_LIBRARY_PATH so the dynamic linker
# searches them before falling back to "file not found".
RUN NVCC_LIB=/opt/conda/lib/python3.11/site-packages/nvidia/cuda_nvcc/nvvm/lib64 && \
    RT_LIB=/opt/conda/lib/python3.11/site-packages/nvidia/cuda_runtime/lib && \
    NVRTC_LIB=/opt/conda/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib && \
    ln -sf "${RT_LIB}/libcudart.so.12"    "${RT_LIB}/libcudart.so" && \
    ln -sf "${NVRTC_LIB}/libnvrtc.so.12"  "${NVRTC_LIB}/libnvrtc.so" && \
    ln -sf "${NVCC_LIB}/libnvvm.so"       "${NVCC_LIB}/libnvvm.so.4"

ENV LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:\
/opt/conda/lib/python3.11/site-packages/nvidia/cuda_nvcc/nvvm/lib64:\
/opt/conda/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:\
/opt/conda/lib/python3.11/site-packages/nvidia/cublas/lib:\
/opt/conda/lib/python3.11/site-packages/nvidia/cusolver/lib:\
/opt/conda/lib/python3.11/site-packages/nvidia/nvjitlink/lib

COPY ./src ./src

EXPOSE 8001
CMD ["uvicorn", "hnn_gpu_service:app", "--host", "0.0.0.0", "--port", "8001"]
