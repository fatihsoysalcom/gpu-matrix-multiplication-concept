import numpy as np
import time

# Define matrix dimensions. For a noticeable difference, these should be reasonably large.
# H100/B200 GPUs handle matrices orders of magnitude larger and faster.
MATRIX_SIZE = 1024 # A common power of 2, good for demonstrating scale

print(f"Demonstrating matrix multiplication, a core operation in AI/Deep Learning.")
print(f"Using matrices of size {MATRIX_SIZE}x{MATRIX_SIZE}.")
print(f"This operation is highly parallelizable, which is why GPUs like H100/B200 excel at it.")
print("-" * 60)

# Create two large random matrices
# Using float32 is common in deep learning for memory and speed efficiency
matrix_a = np.random.rand(MATRIX_SIZE, MATRIX_SIZE).astype(np.float32)
matrix_b = np.random.rand(MATRIX_SIZE, MATRIX_SIZE).astype(np.float32)

print("Matrices created. Starting CPU-bound multiplication...")

# --- CPU-bound Matrix Multiplication using NumPy ---
# NumPy's np.dot is highly optimized for CPUs, often leveraging BLAS libraries.
# However, it still primarily runs sequentially on a few CPU cores.
start_time_cpu = time.perf_counter()
result_matrix_cpu = np.dot(matrix_a, matrix_b)
end_time_cpu = time.perf_counter()

cpu_duration = end_time_cpu - start_time_cpu
print(f"CPU (NumPy) multiplication completed in: {cpu_duration:.4f} seconds.")

print("-" * 60)
print("Concept of GPU acceleration:")
# This is where the article's core concept about H100/B200 GPUs comes in.
# GPUs are designed with thousands of smaller cores for massive parallel computation.
print("While NumPy's `np.dot` is optimized for CPUs, it primarily uses a few CPU cores.")
print("GPUs like NVIDIA H100/B200 have thousands of smaller cores designed for parallel execution.")
print("They can perform this exact matrix multiplication operation orders of magnitude faster")
print("by distributing the calculations across many cores simultaneously.")
print("This massive parallelization is what makes them indispensable for training large AI models,")
print("especially Large Language Models (LLMs) which rely heavily on such operations.")
print("For example, a similar operation on an H100 could take milliseconds instead of seconds.")
print("-" * 60)
