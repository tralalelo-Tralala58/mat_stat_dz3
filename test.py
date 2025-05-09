import math
import matplotlib.pyplot as plt
from collections import *
from scipy.stats import norm, chi2
import csv


def laplace(x):
    """Функция Лапласа (нормированная)"""
    return norm.cdf(x)


data = []

with open('data.csv', 'r') as f:
    for line in f:
        data.append(int(line.strip()))

counter = sorted(Counter(data).items())

n = len(data)
__m = sum(data)/n
__D = sum((x - __m)**2 for x in data)/n

max_elem = max(data)
min_elem = min(data)

k = int(math.floor(math.log2(n))) + 1

h = (max_elem - min_elem)/k

intervals = []
current = min_elem

for _ in range(k):
    next_val = current + h
    intervals.append((current, next_val))
    current = next_val

freq = defaultdict(float)
boundary_values = []

for x in data:
    placed = False
    # Проверяем все интервалы, кроме последнего
    for i, (start, end) in enumerate(intervals[:-1]):
        if math.isclose(x, start):
            freq[i] += 0.5
            if i > 0:
                freq[i - 1] += 0.5
            placed = True
            break
        elif math.isclose(x, end):
            freq[i] += 0.5
            freq[i + 1] += 0.5
            placed = True
            break
        elif start < x < end:
            freq[i] += 1
            placed = True
            break

    # Проверяем последний интервал отдельно
    if not placed:
        start, end = intervals[-1]
        if math.isclose(x, start):
            freq[k - 1] += 0.5
            freq[k - 2] += 0.5
        elif math.isclose(x, end):
            freq[k - 1] += 1  # Последнюю границу включаем полностью
        elif start <= x <= end:
            freq[k - 1] += 1

# Нормализация частот
total = sum(freq.values())
if not math.isclose(total, n):
    correction_factor = n / total
    for key in freq:
        freq[key] *= correction_factor

midpoints = [(start + end)/2 for start, end in intervals]

m_tilde = sum(midpoints[i] * freq[i] for i in freq)/n
D_tilde = sum((midpoints[i] - m_tilde)**2 * freq[i] for i in freq)/n
sigma_tilde = math.sqrt(D_tilde)

plt.figure(figsize=(12, 6))
bars = plt.bar(
    [f"[{start:.1f}-{end:.1f})" for start, end in intervals],
    [freq[i] for i in range(k)],
    width=0.9
)

plt.title(f"Гистограмма распределения роста (n={n}, k={k})")
plt.xlabel("Интервалы роста, см")
plt.ylabel("Частота")
plt.xticks(rotation=45)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f"{height}",
             ha='center', va='bottom')
plt.tight_layout()
plt.savefig('histogram.png')
plt.close()

# Расчет относительных частот
pi = {i: freq[i] / n for i in freq}

p_i = []
for start, end in intervals:
    z1 = (start - m_tilde)/sigma_tilde
    z2 = (end - m_tilde)/sigma_tilde
    p = laplace(z2) - laplace(z1)
    p_i.append(p)

# Относительные частоты p_i*
p_i_star = [freq[i]/n for i in range(k)]

# Квадраты отклонений
squared_deviations = [(p_i_star[i] - p_i[i])**2 for i in range(k)]

i = 0
while i < len(p_i):
    if n * p_i[i] < 5:
        if i < len(p_i) - 1:  # Объединяем с соседним
            p_i[i+1] += p_i[i]
            freq[i+1] += freq[i]
            del p_i[i]
            del freq[i]
            intervals[i] = (intervals[i][0], intervals[i+1][1])
            del intervals[i+1]
        else:  # Последний интервал - объединяем с предыдущим
            p_i[i-1] += p_i[i]
            freq[i-1] += freq[i]
            del p_i[i]
            del freq[i]
            intervals[i-1] = (intervals[i-1][0], intervals[i][1])
            del intervals[i]
    else:
        i += 1

k = len(p_i)  # Обновляем количество интервалов после объединения

# Вычисление статистики χ² (меры расхождения U)
U = sum((freq[i] - n*p_i[i])**2 / (n*p_i[i]) for i in range(k))

# Параметры критерия
s = 2  # Оценивали 2 параметра: m_tilde и sigma_tilde
df = k - 1 - s  # Степени свободы

# Критическое значение (α=0.05)
alpha = 0.05
chi2_critical = chi2.ppf(1 - alpha, df)

with open('output.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=';')

    writer.writerow(["Рост x", "частота f"])
    for height, count in counter:
        writer.writerow([height, count])

    writer.writerow([])

    writer.writerow(["Число опытов (n)", n])
    writer.writerow(["Стат мат ожид", __m])
    writer.writerow(['Стат дисперсия', __D])

    writer.writerow([])
    writer.writerow(['Число разрядов по правилу Стержеса', k])

    writer.writerow([])

    writer.writerow(['Ширина интервала', h])

    writer.writerow([])

    writer.writerow(["Интервал", "mi", "pi", "Сумма mi"])

    cum_sum = 0
    for i in sorted(freq.keys()):
        start, end = intervals[i]
        mi = freq[i]
        cum_sum += mi
        writer.writerow([
            f"[{start:.2f}; {end:.2f})",
            f"{mi:.2f}".replace('.', ','),
            f"{pi[i]:.6f}".replace('.', ','),
            f"{cum_sum:.2f}".replace('.', ',')
        ])

    # Проверки
    writer.writerow([])

    writer.writerow(["Сумма mi", f"{sum(freq.values()):.2f}"])
    writer.writerow(["Сумма pi", f"{sum(pi.values()):.6f}"])

    writer.writerow([])

    writer.writerow(["Интервал", "середина i разряда (xi~)", "(mi~)", "(pi~)"])
    for i in range(k):
        start, end = intervals[i]
        writer.writerow([
            f"[{start:.2f}; {end:.2f})",
            f"{midpoints[i]:.2f}",
            int(freq[i]),
            f"{freq[i] / n:.6f}"
        ])

    writer.writerow([])
    writer.writerow(["m~ (expected value)", f"{m_tilde:.4f}".replace('.', ',')])
    writer.writerow(["D~ (variance)", f"{D_tilde:.4f}".replace('.', ',')])
    writer.writerow(["sigma~ (std deviation)", f"{sigma_tilde:.4f}".replace('.', ',')])
    writer.writerow([])

    writer.writerow(["Интервал", "p_i (теор.)", "p_i* (набл.)", "(p_i* - p_i)^2"])
    for i in range(k):
        start, end = intervals[i]
        writer.writerow([
            f"[{start:.2f}; {end:.2f})",
            f"{p_i[i]:.6f}".replace('.', ','),
            f"{p_i_star[i]:.6f}".replace('.', ','),
            f"{squared_deviations[i]:.6f}".replace('.', ',')
        ])

    writer.writerow([])
    writer.writerow(["Мера расхождения U", f"{U:.4f}".replace('.', ',')])
    writer.writerow(["Критическое значение chi^2:", f"{chi2_critical:.4f}".replace('.', ',')])
    writer.writerow([])
    writer.writerow(["Уровень значимости alpha:", alpha])
    writer.writerow(["Степени свободы df:", df])
    writer.writerow([])

    if U > chi2_critical:
        writer.writerow(["Вывод: отвергаем H0 (распределение не нормальное)"])
    else:
        writer.writerow(["Вывод: нет оснований отвергать H0 (распределение нормальное)"])