import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.io import wavfile

fs, y = wavfile.read('MUHA.wav')
if len(y.shape) > 1:
  y = y[:, 0]
y = y / np.max(np.abs(y))
N = len(y)
dt = 1 / fs
t = np.arange(0, N * dt, dt)
v = np.fft.fftshift(np.fft.fftfreq(N, dt))

# Преобразование Фурье
Y = np.fft.fftshift(np.fft.fft(y)) / N

# Полосовой фильтр
f_low = 300
f_high = 5500
filter_mask = (np.abs(v) >= f_low) & (np.abs(v) <= f_high)
Y_filtered = Y * filter_mask

# Обратное преобразование
y_filtered = np.fft.ifft(np.fft.ifftshift(Y_filtered)) * N
y_filtered = np.real(y_filtered)
y_filtered = y_filtered / np.max(np.abs(y_filtered))

# Воспроизведение
# print("Исходный звук...")
# sd.play(y, fs)
# sd.wait()
# print("Отфильтрованный звук...")
# sd.play(y_filtered, fs)
# sd.wait()

# Графики
plt.figure(figsize=(10, 10))
plt.subplot(2, 1, 1)
plt.plot(t, y, label='Исходный сигнал', alpha=0.7)
plt.plot(t, y_filtered, label='Фильтрованный сигнал', color='green')
plt.xlabel('Время (с)')
plt.ylabel('Амплитуда')
plt.title('Сигналы во временной области')
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(v, np.abs(Y), label='Исходный Фурье-образ', alpha=0.7)
plt.plot(v, np.abs(Y_filtered), label='Фильтрованный Фурье-образ', color='green')
plt.axvspan(-f_high, -f_low, color='gray', alpha=0.3, label='Проходная полоса')
plt.axvspan(f_low, f_high, color='gray', alpha=0.3)
plt.xlabel('Частота (Гц)')
plt.ylabel('Амплитуда')
plt.title('Сигналы в частотной области (модули преобразований Фурье)')
plt.xlim(-6000, 6000)
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

wavfile.write('MUHA_filtered.wav', fs, y_filtered.astype(np.float32))
