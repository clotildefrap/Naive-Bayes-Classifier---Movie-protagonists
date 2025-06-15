import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import shapiro

# Accuracy scores
accuracy_function = [0.4081, 0.4611, 0.5296, 0.5447, 0.5618, 0.5883, 0.5644, 0.5337]
accuracy_content  = [0.5613, 0.5670, 0.5966, 0.6012, 0.5363, 0.3738, 0.1833, 0.1251]

# Shapiro-Wilk test for normality

diffs = np.array(accuracy_content) - np.array(accuracy_function)


stat, p = shapiro(diffs)
print(f"Shapiro-Wilk test: statistic = {stat:.4f}, p-value = {p:.4f}")


# Run paired t-test
t_stat, p_value = stats.ttest_rel(accuracy_content, accuracy_function)

print(f"Paired t-test results:")
print(f"t-statistic = {t_stat:.4f}")
print(f"p-value     = {p_value:.4f}")
