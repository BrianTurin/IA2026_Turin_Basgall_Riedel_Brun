import random
import time

def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]

sizes = [1024, 4096, 16384, 131072]

for size in sizes:
    arr = [random.randint(0, 100000) for _ in range(size)]
    print(f"Ejecutando BubbleSort con tamaño {size}...")
    start = time.time()
    bubble_sort(arr)
    end = time.time()
    print(f"Tiempo de ejecución (BubbleSort, {size}): {end - start:.4f} segundos\n")
