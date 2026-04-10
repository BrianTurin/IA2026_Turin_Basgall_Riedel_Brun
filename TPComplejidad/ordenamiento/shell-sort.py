import random
import time

def shell_sort(arr):
    n = len(arr)
    gap = n // 2
    while gap > 0:
        for i in range(gap, n):
            temp = arr[i]
            j = i
            while j >= gap and arr[j - gap] > temp:
                arr[j] = arr[j - gap]
                j -= gap
            arr[j] = temp
        gap //= 2

sizes = [1024, 4096, 16384, 131072]

for size in sizes:
    arr = [random.randint(0, 100000) for _ in range(size)]
    print(f"Ejecutando ShellSort con tamaño {size}...")
    start = time.time()
    shell_sort(arr)
    end = time.time()
    print(f"Tiempo de ejecución (ShellSort, {size}): {end - start:.4f} segundos\n")
