import librosa
import numpy as np
import matplotlib.pyplot as plt

file_path = "аккорд_13.mp3"
data, sample_rate = librosa.load(file_path, sr=None, mono=True)

# График временного сигнала
t = np.arange(len(data)) / sample_rate
plt.figure(figsize=(12, 4))
plt.plot(t, data)
plt.title("График f(t)")
plt.xlabel("t, с")
plt.ylabel("f(t)")
plt.grid()
plt.show()


def compute_fourier(signal, t, freq_range):
    result = []
    dt = t[1] - t[0]  # шаг по времени
    for freq in freq_range:
        integrand = signal * np.exp(-1j * 2 * np.pi * freq * t)
        result.append(np.sum(integrand) * dt)  # метод прямоугольников
    return np.array(result)


V = 800  # Максимальная частота (Гц)
df = 1.0  # Шаг по частоте (Гц)
freq_range = np.arange(80, V + df, df)  # Исключаем очень низкие частоты (ниже 80 Гц)

fourier = compute_fourier(data, t, freq_range)
amplitude = np.abs(fourier)

# График спектра
plt.figure(figsize=(12, 4))
plt.plot(freq_range, amplitude)
plt.title("График |f(v)|")
plt.xlabel("Частота ω, Гц")
plt.ylabel("|f(v)|")
plt.grid()
plt.show()


def find_main_peaks(spectrum, freqs):
    spectrum = spectrum.copy()
    peaks = []

    for _ in range(3):  # Ищем 3 пика (так как аккорд состоит их трёх нот)
        max_idx = np.argmax(spectrum)
        max_freq = freqs[max_idx]
        peaks.append(max_freq)

        # Обнуляем область вокруг найденного пика (чтобы не было лишних "нот" рядом)
        mask = (freqs > max_freq - 40) & (freqs < max_freq + 40)
        spectrum[mask] = 0

    return sorted(peaks)


main_freqs = find_main_peaks(amplitude, freq_range)


# Преобразуем частоты в ноты
def freq_to_note(freq):
    note_freqs = {
        'A': [27.50, 55.00, 110.00, 220.00, 440.00, 880.00],
        'A#': [29.14, 58.27, 116.54, 233.08, 466.16, 932.33],
        'B': [30.87, 61.74, 123.47, 246.94, 493.88, 987.77],
        'C': [16.35, 32.70, 65.41, 130.81, 261.63, 523.25],
        'C#': [17.32, 34.65, 69.30, 138.59, 277.18, 554.37],
        'D': [18.35, 36.71, 73.42, 146.83, 293.66, 587.33],
        'D#': [19.45, 38.89, 77.78, 155.56, 311.13, 622.25],
        'E': [20.60, 41.20, 82.41, 164.81, 329.63, 659.25],
        'F': [21.83, 43.65, 87.31, 174.61, 349.23, 698.46],
        'F#': [23.12, 46.25, 92.50, 185.00, 369.99, 739.99],
        'G': [24.50, 49.00, 98.00, 196.00, 392.00, 783.99],
        'G#': [25.96, 51.91, 103.83, 207.65, 415.30, 830.61]
    }

    closest_note = None
    min_diff = float('inf')

    for note, freqs_list in note_freqs.items():
        for octave, note_freq in enumerate(freqs_list):
            diff = abs(freq - note_freq)
            if diff < min_diff:
                min_diff = diff
                closest_note = f"{note}{octave}"

    return closest_note


print("Основные ноты аккорда:")
notes = [freq_to_note(f) for f in main_freqs]
for freq, note in zip(main_freqs, notes):
    print(f"{freq:.2f} Гц ~ {note}")

plt.figure(figsize=(12, 4))
plt.plot(freq_range, amplitude)
plt.scatter(main_freqs, [amplitude[np.where(freq_range == f)[0][0]] for f in main_freqs],
            color='red', s=100, label="Основные ноты")
for freq, note in zip(main_freqs, notes):
    plt.annotate(f"{note}\n{freq:.1f} Гц", (freq, amplitude[np.where(freq_range == freq)[0][0]]),
                 textcoords="offset points", xytext=(0, 10), ha='center', fontsize=12)
plt.title("График |f(v)|")
plt.xlabel("Частота ω, Гц")
plt.ylabel("|f(v)|")
plt.xlim(200, 800)
plt.grid()
plt.legend()
plt.show()
