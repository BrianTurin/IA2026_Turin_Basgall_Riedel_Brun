import random
import time

def quick_sort(arr):
    def _quick_sort(low, high):
        if low < high:
            pi = partition(low, high)
            _quick_sort(low, pi - 1)
            _quick_sort(pi + 1, high)

    def partition(low, high):
        pivot = arr[high]
        i = low - 1
        for j in range(low, high):
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        return i + 1

    _quick_sort(0, len(arr) - 1)

sizes = [1024, 4096, 16384, 131072]

for size in sizes:
    arr = [random.randint(0, 100000) for _ in range(size)]
    print(f"Ejecutando QuickSort con tamaño {size}...")
    start = time.time()
    quick_sort(arr)
    end = time.time()
    print(f"Tiempo de ejecución (QuickSort, {size}): {end - start:.4f} segundos\n")
