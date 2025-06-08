import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv2
from scipy.ndimage import convolve


# Задание 1

# Получение Фурье-образа
img = plt.imread('./plots/task1/12.png').astype(np.float64)

img_fourier = np.stack([np.fft.fftshift(np.fft.fft2(img[:, :, i])) for i in range(3)], axis=2)

abs_fourier = np.abs(img_fourier)
phase_fourier = np.angle(img_fourier)

abs_fourier_log = np.log(abs_fourier + 1)
abs_fourier_log_norm = (np.log(abs_fourier + 1) - np.min(abs_fourier_log)) / (np.max(abs_fourier_log) - np.min(abs_fourier_log))

plt.imsave('./plots/task1/log_fft.png', abs_fourier_log_norm, cmap='gray')

# Восстановление изображения
abs_fourier_log_norm = plt.imread('./plots/task1/log_fft_filtered.png')

abs_fourier = np.exp(abs_fourier_log_norm * (np.max(abs_fourier_log) - np.min(abs_fourier_log)) + np.min(abs_fourier_log))
img_fourier = abs_fourier * np.exp(1j * phase_fourier)

img_restored = np.clip(np.stack([abs(np.fft.ifft2(np.fft.ifftshift(img_fourier[:, :, i]))) for i in range(3)], axis=2), 0, 1)

plt.imsave('./plots/task1/restored.png', img_restored, cmap='gray')

# Загрузка исходного изображения
img = plt.imread('./plots/task1/12.png').astype(np.float64)

# Вычисление Фурье-образа
img_fourier = np.stack([np.fft.fftshift(np.fft.fft2(img[:, :, i])) for i in range(3)], axis=2)
abs_fourier = np.abs(img_fourier)
phase_fourier = np.angle(img_fourier)

abs_fourier_log = np.log(abs_fourier + 1)
abs_fourier_log_norm = (np.log(abs_fourier + 1) - np.min(abs_fourier_log)) / (np.max(abs_fourier_log) - np.min(abs_fourier_log))
plt.imsave('./plots/task1/log_fft.png', abs_fourier_log_norm, cmap='gray')

# Восстановление изображения
abs_fourier_log_norm_filtered = plt.imread('./plots/task1/log_fft_filtered.png')

abs_fourier = np.exp(abs_fourier_log_norm_filtered * (np.max(abs_fourier_log) - np.min(abs_fourier_log)) + np.min(abs_fourier_log))
img_fourier = abs_fourier * np.exp(1j * phase_fourier)

img_restored = np.clip(np.stack([abs(np.fft.ifft2(np.fft.ifftshift(img_fourier[:, :, i]))) for i in range(3)], axis=2), 0, 1)

# График сравнения исходного и восстановленного изображений
plt.figure(figsize=(12, 8))
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.title('Исходное изображение')

plt.subplot(1, 2, 2)
plt.imshow(img_restored, cmap='gray')
plt.axis('off')
plt.title('Восстановленное изображение')
plt.tight_layout()
plt.savefig('./plots/task1/comparison.png', bbox_inches='tight', dpi=300)
plt.close()

# График Фурье-образов
plt.figure(figsize=(12, 8))
plt.subplot(1, 2, 1)
plt.imshow(abs_fourier_log_norm, cmap='gray')
plt.axis('off')
plt.title('Исходный Фурье-образ')

plt.subplot(1, 2, 2)
plt.imshow(abs_fourier_log_norm_filtered, cmap='gray')
plt.axis('off')
plt.title('Фильтрованный Фурье-образ')
plt.tight_layout()
plt.savefig('./plots/task1/comparison_fourier.png', bbox_inches='tight', dpi=300)
plt.close()

# График Фурье-образа: исходный и фильтрованный

height, width = abs_fourier_log_norm.shape[:2]
center_y, center_x = height // 2, width // 2

# Срезы для центральной области
cropped_abs = abs_fourier_log_norm[center_y - 150:center_y + 150, center_x - 150:center_x + 150]
cropped_filtered = abs_fourier_log_norm_filtered[center_y - 150:center_y + 150, center_x - 150:center_x + 150]

plt.figure(figsize=(12, 8))
plt.subplot(1, 2, 1)
plt.imshow(cropped_abs, cmap='gray')
plt.axis('off')
plt.title('Исходный Фурье-образ (центр)')

plt.subplot(1, 2, 2)
plt.imshow(cropped_filtered, cmap='gray')
plt.axis('off')
plt.title('Фильтрованный Фурье-образ (центр)')

plt.tight_layout()
plt.savefig('./plots/task1/comparison_fourier_cropp.png', bbox_inches='tight', dpi=300)
plt.close()


# Задание 2

def gauss(img, N):
  path = './plots/task2/gauss/'
  for n in N:
    j, i = np.meshgrid(np.arange(n), np.arange(n))
    kernel = np.exp(-((i - (n + 1) / 2) ** 2 + (j - (n + 1) / 2) ** 2) / (2 * ((n - 1) / 6) ** 2))
    kernel /= kernel.sum()

    imv_conv = convolve(img, kernel)

    h, w = img.shape
    k, l = kernel.shape
    padded_img = np.pad(img, ((0, k - 1), (0, l - 1)), mode='constant')
    padded_kernel = np.pad(kernel, ((0, h - k), (0, w - l)), mode='constant')

    img_fourier = np.fft.fft2(img)
    kernel_fourier = np.fft.fft2(padded_kernel)

    abs_fourier = np.abs(np.fft.fftshift(img_fourier))
    abs_fourier_log = np.log(abs_fourier + 1)
    abs_fourier_log_norm_fourier = (np.log(abs_fourier + 1) - np.min(abs_fourier_log)) / (
        np.max(abs_fourier_log) - np.min(abs_fourier_log))

    abs_fourier = np.abs(np.fft.fftshift(kernel_fourier))
    abs_fourier_log = np.log(abs_fourier + 1)
    abs_fourier_log_norm_kernel = (np.log(abs_fourier + 1) - np.min(abs_fourier_log)) / (
        np.max(abs_fourier_log) - np.min(abs_fourier_log))

    img_fft_mult = img_fourier * kernel_fourier

    abs_fourier = np.abs(np.fft.fftshift(img_fft_mult))
    abs_fourier_log = np.log(abs_fourier + 1)
    abs_fourier_log_norm_res = (np.log(abs_fourier + 1) - np.min(abs_fourier_log)) / (
        np.max(abs_fourier_log) - np.min(abs_fourier_log))

    img_fft = np.fft.ifft2(img_fft_mult).real

    plt.figure(figsize=(12, 8))
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title('Исходное изображение')

    plt.subplot(1, 3, 2)
    plt.imshow(imv_conv, cmap='gray')
    plt.axis('off')
    plt.title(f'Результат свертки N = {n}')

    plt.subplot(1, 3, 3)
    plt.imshow(img_fft, cmap='gray')
    plt.axis('off')
    plt.title('Результат Фурье-образов')

    plt.tight_layout()
    plt.savefig(path + f'/cmp_img_{n}.png', bbox_inches='tight', dpi=300)
    plt.close()

    plt.figure(figsize=(12, 8))
    plt.subplot(1, 3, 1)
    plt.imshow(abs_fourier_log_norm_fourier, cmap='gray')
    plt.axis('off')
    plt.title('Исходный Фурье-образ')

    plt.subplot(1, 3, 2)
    plt.imshow(abs_fourier_log_norm_kernel, cmap='gray')
    plt.axis('off')
    plt.title(f'Фурье-образ ядра N = {n}')

    plt.subplot(1, 3, 3)
    plt.imshow(abs_fourier_log_norm_res, cmap='gray')
    plt.axis('off')
    plt.title('Полученный Фурье-образ')

    plt.tight_layout()
    plt.savefig(path + f'/cmp_furr_{n}.png', bbox_inches='tight', dpi=300)
    plt.close()


def block(img, N):
  path = './plots/task2/block/'
  for n in N:
    kernel = np.ones((n, n))
    kernel /= kernel.sum()

    imv_conv = convolve(img, kernel)

    h, w = img.shape
    k, l = kernel.shape
    padded_img = np.pad(img, ((0, k - 1), (0, l - 1)), mode='constant')
    padded_kernel = np.pad(kernel, ((0, h - k), (0, w - l)), mode='constant')

    img_fourier = np.fft.fft2(img)
    kernel_fourier = np.fft.fft2(padded_kernel)

    abs_fourier = np.abs(np.fft.fftshift(img_fourier))
    abs_fourier_log = np.log(abs_fourier + 1)
    abs_fourier_log_norm_fourier = (np.log(abs_fourier + 1) - np.min(abs_fourier_log)) / (
        np.max(abs_fourier_log) - np.min(abs_fourier_log))

    abs_fourier = np.abs(np.fft.fftshift(kernel_fourier))
    abs_fourier_log = np.log(abs_fourier + 1)
    abs_fourier_log_norm_kernel = (np.log(abs_fourier + 1) - np.min(abs_fourier_log)) / (
        np.max(abs_fourier_log) - np.min(abs_fourier_log))

    img_fft_mult = img_fourier * kernel_fourier

    abs_fourier = np.abs(np.fft.fftshift(img_fft_mult))
    abs_fourier_log = np.log(abs_fourier + 1)
    abs_fourier_log_norm_res = (np.log(abs_fourier + 1) - np.min(abs_fourier_log)) / (
        np.max(abs_fourier_log) - np.min(abs_fourier_log))

    img_fft = np.fft.ifft2(img_fft_mult).real

    plt.figure(figsize=(12, 8))
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title('Исходное изображение')

    plt.subplot(1, 3, 2)
    plt.imshow(imv_conv, cmap='gray')
    plt.axis('off')
    plt.title(f'Результат свертки N = {n}')

    plt.subplot(1, 3, 3)
    plt.imshow(img_fft, cmap='gray')
    plt.axis('off')
    plt.title('Результат Фурье-образов')

    plt.tight_layout()
    plt.savefig(path + f'/cmp_img_{n}.png', bbox_inches='tight', dpi=300)
    plt.close()

    plt.figure(figsize=(12, 8))
    plt.subplot(1, 3, 1)
    plt.imshow(abs_fourier_log_norm_fourier, cmap='gray')
    plt.axis('off')
    plt.title('Исходный Фурье-образ')

    plt.subplot(1, 3, 2)
    plt.imshow(abs_fourier_log_norm_kernel, cmap='gray')
    plt.axis('off')
    plt.title(f'Фурье-образ ядра N = {n}')

    plt.subplot(1, 3, 3)
    plt.imshow(abs_fourier_log_norm_res, cmap='gray')
    plt.axis('off')
    plt.title('Полученный Фурье-образ')

    plt.tight_layout()
    plt.savefig(path + f'/cmp_furr_{n}.png', bbox_inches='tight', dpi=300)
    plt.close()


def sharp(img):
  path = './plots/task2/sharp/'
  kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

  imv_conv = convolve(img, kernel)

  h, w = img.shape
  k, l = kernel.shape
  padded_img = np.pad(img, ((0, k - 1), (0, l - 1)), mode='constant')
  padded_kernel = np.pad(kernel, ((0, h - k), (0, w - l)), mode='constant')

  img_fourier = np.fft.fft2(img)
  kernel_fourier = np.fft.fft2(padded_kernel)

  abs_fourier = np.abs(np.fft.fftshift(img_fourier))
  abs_fourier_log = np.log(abs_fourier + 1)
  abs_fourier_log_norm_fourier = (np.log(abs_fourier + 1) - np.min(abs_fourier_log)) / (
      np.max(abs_fourier_log) - np.min(abs_fourier_log))

  abs_fourier = np.abs(np.fft.fftshift(kernel_fourier))
  abs_fourier_log = np.log(abs_fourier + 1)
  abs_fourier_log_norm_kernel = (np.log(abs_fourier + 1) - np.min(abs_fourier_log)) / (
      np.max(abs_fourier_log) - np.min(abs_fourier_log))

  img_fft_mult = img_fourier * kernel_fourier

  abs_fourier = np.abs(np.fft.fftshift(img_fft_mult))
  abs_fourier_log = np.log(abs_fourier + 1)
  abs_fourier_log_norm_res = (np.log(abs_fourier + 1) - np.min(abs_fourier_log)) / (
      np.max(abs_fourier_log) - np.min(abs_fourier_log))

  img_fft = np.fft.ifft2(img_fft_mult).real

  plt.figure(figsize=(12, 8))
  plt.subplot(1, 3, 1)
  plt.imshow(img, cmap='gray')
  plt.axis('off')
  plt.title('Исходное изображение')

  plt.subplot(1, 3, 2)
  plt.imshow(imv_conv, cmap='gray')
  plt.axis('off')
  plt.title(f'Результат свертки')

  plt.subplot(1, 3, 3)
  plt.imshow(imv_conv, cmap='gray')
  plt.axis('off')
  plt.title('Результат Фурье-образов')

  plt.tight_layout()
  plt.savefig(path + f'/cmp_img.png', bbox_inches='tight', dpi=300)
  plt.close()

  plt.figure(figsize=(12, 8))
  plt.subplot(1, 3, 1)
  plt.imshow(abs_fourier_log_norm_fourier, cmap='gray')
  plt.axis('off')
  plt.title('Исходный Фурье-образ')

  plt.subplot(1, 3, 2)
  plt.imshow(abs_fourier_log_norm_kernel, cmap='gray')
  plt.axis('off')
  plt.title(f'Фурье-образ ядра')

  plt.subplot(1, 3, 3)
  plt.imshow(abs_fourier_log_norm_res, cmap='gray')
  plt.axis('off')
  plt.title('Полученный Фурье-образ')

  plt.tight_layout()
  plt.savefig(path + f'/cmp_furr.png', bbox_inches='tight', dpi=300)
  plt.close()


def edges(img):
  path = './plots/task2/edges/'
  kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

  imv_conv = convolve(img, kernel)

  h, w = img.shape
  k, l = kernel.shape
  padded_img = np.pad(img, ((0, k - 1), (0, l - 1)), mode='constant')
  padded_kernel = np.pad(kernel, ((0, h - k), (0, w - l)), mode='constant')

  img_fourier = np.fft.fft2(img)
  kernel_fourier = np.fft.fft2(padded_kernel)

  abs_fourier = np.abs(np.fft.fftshift(img_fourier))
  abs_fourier_log = np.log(abs_fourier + 1)
  abs_fourier_log_norm_fourier = (np.log(abs_fourier + 1) - np.min(abs_fourier_log)) / (
      np.max(abs_fourier_log) - np.min(abs_fourier_log))

  abs_fourier = np.abs(np.fft.fftshift(kernel_fourier))
  abs_fourier_log = np.log(abs_fourier + 1)
  abs_fourier_log_norm_kernel = (np.log(abs_fourier + 1) - np.min(abs_fourier_log)) / (
      np.max(abs_fourier_log) - np.min(abs_fourier_log))

  img_fft_mult = img_fourier * kernel_fourier

  abs_fourier = np.abs(np.fft.fftshift(img_fft_mult))
  abs_fourier_log = np.log(abs_fourier + 1)
  abs_fourier_log_norm_res = (np.log(abs_fourier + 1) - np.min(abs_fourier_log)) / (
      np.max(abs_fourier_log) - np.min(abs_fourier_log))

  img_fft = np.fft.ifft2(img_fft_mult).real

  plt.figure(figsize=(12, 8))
  plt.subplot(1, 3, 1)
  plt.imshow(img, cmap='gray')
  plt.axis('off')
  plt.title('Исходное изображение')

  plt.subplot(1, 3, 2)
  plt.imshow(imv_conv, cmap='gray')
  plt.axis('off')
  plt.title(f'Результат свертки')

  plt.subplot(1, 3, 3)
  plt.imshow(imv_conv, cmap='gray')
  plt.axis('off')
  plt.title('Результат Фурье-образов')

  plt.tight_layout()
  plt.savefig(path + f'/cmp_img.png', bbox_inches='tight', dpi=300)
  plt.close()

  plt.figure(figsize=(12, 8))
  plt.subplot(1, 3, 1)
  plt.imshow(abs_fourier_log_norm_fourier, cmap='gray')
  plt.axis('off')
  plt.title('Исходный Фурье-образ')

  plt.subplot(1, 3, 2)
  plt.imshow(abs_fourier_log_norm_kernel, cmap='gray')
  plt.axis('off')
  plt.title(f'Фурье-образ ядра')

  plt.subplot(1, 3, 3)
  plt.imshow(abs_fourier_log_norm_res, cmap='gray')
  plt.axis('off')
  plt.title('Полученный Фурье-образ')

  plt.tight_layout()
  plt.savefig(path + f'/cmp_furr.png', bbox_inches='tight', dpi=300)
  plt.close()


def embos(img):
  path = './plots/task2/embos/'
  kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])

  imv_conv = convolve(img, kernel)

  h, w = img.shape
  k, l = kernel.shape
  padded_img = np.pad(img, ((0, k - 1), (0, l - 1)), mode='constant')
  padded_kernel = np.pad(kernel, ((0, h - k), (0, w - l)), mode='constant')

  img_fourier = np.fft.fft2(img)
  kernel_fourier = np.fft.fft2(padded_kernel)

  abs_fourier = np.abs(np.fft.fftshift(img_fourier))
  abs_fourier_log = np.log(abs_fourier + 1)
  abs_fourier_log_norm_fourier = (np.log(abs_fourier + 1) - np.min(abs_fourier_log)) / (
      np.max(abs_fourier_log) - np.min(abs_fourier_log))

  abs_fourier = np.abs(np.fft.fftshift(kernel_fourier))
  abs_fourier_log = np.log(abs_fourier + 1)
  abs_fourier_log_norm_kernel = (np.log(abs_fourier + 1) - np.min(abs_fourier_log)) / (
      np.max(abs_fourier_log) - np.min(abs_fourier_log))

  img_fft_mult = img_fourier * kernel_fourier

  abs_fourier = np.abs(np.fft.fftshift(img_fft_mult))
  abs_fourier_log = np.log(abs_fourier + 1)
  abs_fourier_log_norm_res = (np.log(abs_fourier + 1) - np.min(abs_fourier_log)) / (
      np.max(abs_fourier_log) - np.min(abs_fourier_log))

  img_fft = np.fft.ifft2(img_fft_mult).real

  plt.figure(figsize=(12, 8))
  plt.subplot(1, 3, 1)
  plt.imshow(img, cmap='gray')
  plt.axis('off')
  plt.title('Исходное изображение')

  plt.subplot(1, 3, 2)
  plt.imshow(imv_conv, cmap='gray')
  plt.axis('off')
  plt.title(f'Результат свертки')

  plt.subplot(1, 3, 3)
  plt.imshow(imv_conv, cmap='gray')
  plt.axis('off')
  plt.title('Результат Фурье-образов')

  plt.tight_layout()
  plt.savefig(path + f'/cmp_img.png', bbox_inches='tight', dpi=300)
  plt.close()

  plt.figure(figsize=(12, 8))
  plt.subplot(1, 3, 1)
  plt.imshow(abs_fourier_log_norm_fourier, cmap='gray')
  plt.axis('off')
  plt.title('Исходный Фурье-образ')

  plt.subplot(1, 3, 2)
  plt.imshow(abs_fourier_log_norm_kernel, cmap='gray')
  plt.axis('off')
  plt.title(f'Фурье-образ ядра')

  plt.subplot(1, 3, 3)
  plt.imshow(abs_fourier_log_norm_res, cmap='gray')
  plt.axis('off')
  plt.title('Полученный Фурье-образ')

  plt.tight_layout()
  plt.savefig(path + f'/cmp_furr.png', bbox_inches='tight', dpi=300)
  plt.close()


img = cv2.cvtColor(plt.imread('./plots/task2/original.jpg'), cv2.COLOR_BGR2GRAY)
plt.imsave('./plots/task2/original_gray.png', img, cmap='gray')

N = [10, 30, 60]
# gauss(img, N)
# block(img, N)
# sharp(img)
edges(img)
embos(img)

