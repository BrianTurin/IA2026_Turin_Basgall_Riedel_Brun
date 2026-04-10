import random
import time

def selection_sort(arr):
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]

sizes = [1024, 4096, 16384, 131072]

for size in sizes:
    arr = [random.randint(0, 100000) for _ in range(size)]
    print(f"Ejecutando SelectionSort con tamaño {size}...")
    start = time.time()
    selection_sort(arr)
    end = time.time()
    print(f"Tiempo de ejecución (SelectionSort, {size}): {end - start:.4f} segundos\n")
