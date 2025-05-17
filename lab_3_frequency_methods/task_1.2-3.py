import numpy as np
import matplotlib.pyplot as plt

T = 20.0
dt = 0.01
t = np.arange(-T / 2, T / 2, dt)
N = len(t)
V = 1 / dt
dv = 1 / T
v = np.arange(-V / 2, V / 2, dv)

a = 2.0
t1, t2 = -2.0, 2.0
b = 0.0
c = 0.5
d = 10.0
g = np.zeros_like(t)
g[(t >= t1) & (t <= t2)] = a
xi = np.random.uniform(-1, 1, size=t.shape)
u = g + b * xi + c * np.sin(d * t)

# Ппреобразование Фурье
U = np.fft.fftshift(np.fft.fft(u)) / N
g_hat = np.fft.fftshift(np.fft.fft(g)) / N

# Фильтр (полосовой + низких частот)
nu_0 = 5.0
delta = 1.0
freq_sine = d / (2 * np.pi)
filter_mask = np.ones_like(v)
filter_mask[(np.abs(v) >= freq_sine - delta) & (np.abs(v) <= freq_sine + delta)] = 0
filter_mask[np.abs(v) > nu_0] = 0
U_filtered = U * filter_mask

# Обратное преобразование Фурье
u_filtered = np.fft.ifft(np.fft.ifftshift(U_filtered)) * N
u_filtered = np.real(u_filtered)

# Графики
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(t, g, label='Исходный g(t)', color='blue')
plt.plot(t, u, label='Зашумлённый u(t)', color='red', alpha=0.5)
plt.plot(t, u_filtered, label='Фильтрованный сигнал', color='green')
plt.xlabel('Время (с)')
plt.ylabel('Амплитуда')
plt.legend()
plt.title('Сигналы во временной области')
plt.grid()

plt.subplot(2, 1, 2)
plt.axvspan(-freq_sine - delta, -freq_sine + delta, color='gray', alpha=0.3, label='Полоса фильтра')
plt.axvspan(freq_sine - delta, freq_sine + delta, color='gray', alpha=0.3)
plt.plot(v, np.abs(g_hat), label='|ĝ(ν)|', color='blue')
plt.plot(v, np.abs(U), label='|û(ν)|', color='red', alpha=0.5)
plt.plot(v, np.abs(U_filtered), label='|Фурье-образ фильтрованного|', color='green')
plt.xlabel('Частота (Гц)')
plt.ylabel('Амплитуда')
plt.legend()
plt.title('Сигналы в частотной области (модули преобразований Фурье)')
plt.xlim(-20, 20)
plt.grid()
plt.tight_layout()
plt.show()
