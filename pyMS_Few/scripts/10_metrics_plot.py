import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
from scipy.optimize import brentq

# 从CSV文件读取数据
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None

# 寻找最佳高斯数量
def _find_optimal_k(deriv):
    N = len(deriv)
    delta = np.zeros(N)
    for i in range(N - 2):
        subsequent_mean = np.mean(deriv[i+1:])
        delta[i] = abs(deriv[i] - subsequent_mean)

    steep_slope_idx = None
    for i in range(N - 3, -1, -1):
        if delta[i] > 0.05:
            steep_slope_idx = i
            break

    if steep_slope_idx is None:
        return 2

    k_opt = steep_slope_idx + 1
    for j in range(steep_slope_idx, -1, -1):
        if delta[j] <= 0.02:
            k_opt = j + 2
            break

    return max(2, k_opt)

# 计算两个高斯函数的交点
def _find_gaussian_intersection(mu1, sigma1, w1, mu2, sigma2, w2):
    try:
        def _gaussian_diff(x):
            return w1 * norm.pdf(x, mu1, sigma1) - w2 * norm.pdf(x, mu2, sigma2)
        intersection = brentq(_gaussian_diff, mu1, mu2)
        return intersection
    except:
        return (mu1 + mu2) / 2  # 数值方法失败时回退

# GMM拟合并找到阈值和μ+3σ
def fit_gmm_and_find_threshold(values, max_components=10):
    values = values.dropna()
    if len(values) < 10:
        print("Not enough data points to fit GMM.")
        return None, None, None

    data = np.array(values).reshape(-1, 1)

    # 动态选择高斯数量
    K_range = range(1, max_components + 1)
    bic_values = []
    for k in K_range:
        gmm = GaussianMixture(n_components=k, random_state=42)
        gmm.fit(data)
        bic_values.append(gmm.bic(data))

    deriv = np.abs(np.diff(bic_values))
    deriv /= np.max(deriv)
    k_opt = _find_optimal_k(deriv)

    # 拟合最终模型
    gmm = GaussianMixture(n_components=k_opt, random_state=42)
    gmm.fit(data)

    # 提取并排序参数
    order = np.argsort(gmm.means_.flatten())
    means = gmm.means_.flatten()[order]
    sigmas = np.sqrt(gmm.covariances_.flatten()[order])
    weights = gmm.weights_[order]

    # 计算第一个高斯的μ+3σ
    mu_plus_3sigma = means[0] + 3 * sigmas[0] if k_opt >= 1 else None

    # 计算前两个高斯的交点
    intersection = None
    if k_opt >= 2:
        intersection = _find_gaussian_intersection(
            means[0], sigmas[0], weights[0],
            means[1], sigmas[1], weights[1]
        )

    return gmm, intersection, mu_plus_3sigma

# 绘制直方图并标注阈值和μ+3σ
def plot_histogram_with_threshold(data, column_name, bins=30):
    if column_name not in data.columns:
        print(f"Column '{column_name}' not found in the data.")
        return None, None

    # 提取指定列的数据
    values = data[column_name]

    # 拟合GMM并找到交点和μ+3σ
    gmm, threshold, mu_plus_3sigma = fit_gmm_and_find_threshold(values)
    if gmm is None or mu_plus_3sigma is None:
        print("Failed to determine threshold or μ+3σ.")
        return None, None

    # 绘制直方图（使用频次而不是密度）
    plt.figure(figsize=(10, 6))
    counts, bins_edges, _ = plt.hist(values, bins=bins, edgecolor='black', alpha=0.7, density=False)

    # 计算bin宽度，用于缩放GMM曲线
    bin_width = bins_edges[1] - bins_edges[0]
    total_counts = len(values)  # 总频次

    # 绘制GMM拟合曲线（缩放到频次）
    x = np.linspace(min(bins_edges), max(bins_edges), 100)
    means = gmm.means_.flatten()
    sigmas = np.sqrt(gmm.covariances_.flatten())
    weights = gmm.weights_
    y_total = np.zeros_like(x)
    for i in range(gmm.n_components):
        y = weights[i] * norm.pdf(x, means[i], sigmas[i])
        y_scaled = y * total_counts * bin_width
        plt.plot(x, y_scaled, label=f'Gaussian {i+1}', linestyle='--')
        y_total += y
    y_total_scaled = y_total * total_counts * bin_width
    plt.plot(x, y_total_scaled, label='GMM Fit', color='red')

    # 标注前两个高斯的交点（阈值）
    if threshold is not None:
        plt.axvline(threshold, color='green', linestyle='--', label=f'Threshold: {threshold:.2f}')

    # 标注第一个高斯的μ+3σ
    plt.axvline(mu_plus_3sigma, color='blue', linestyle='-.', label=f'μ+3σ: {mu_plus_3sigma:.2f}')

    # 设置标题和标签
    plt.title(f'Histogram of {column_name} with GMM, Threshold, and μ+3σ')
    plt.xlabel(column_name)
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 保存图像
    plt.savefig(f'histogram_{column_name}_with_threshold_and_mu3sigma.png')
    print(f"Histogram with threshold and μ+3σ saved as 'histogram_{column_name}_with_threshold_and_mu3sigma.png'")
    plt.close()

    return threshold, mu_plus_3sigma

# 主函数
def main():
    # 文件路径
    file_path = './aligned_clusters_filled_with_metrics.csv'

    # 加载数据
    data = load_data(file_path)
    if data is None:
        return

    # 定义要处理的指标
    columns = ['mcq_index', '1-apex_max_boundary']

    # 保存阈值的字典
    thresholds = {}

    # 自动处理两个指标
    for column_name in columns:
        print(f"Processing column: {column_name}")
        threshold, mu_plus_3sigma = plot_histogram_with_threshold(data, column_name)
        if mu_plus_3sigma is not None:
            thresholds[column_name] = mu_plus_3sigma
        else:
            print(f"Failed to compute μ+3σ for {column_name}")

    # 保存阈值到 thresholds.txt
    with open('thresholds.txt', 'w') as f:
        for column, mu_plus_3sigma in thresholds.items():
            f.write(f"{column}:{mu_plus_3sigma}\n")
    print("Thresholds saved to thresholds.txt")

if __name__ == "__main__":
    main()