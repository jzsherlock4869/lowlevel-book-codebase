import numpy as np
import matplotlib.pyplot as plt

def remapping_tone(x, g0, sigma, beta):
    region = (np.abs(x - g0) > sigma)
    r = g0 + np.sign(x - g0) * (beta * (np.abs(x - g0) - sigma) + sigma)
    remapped = region * r + (1 - region) * x
    return remapped.astype(x.dtype)

def remapping_detail(x, g0, sigma, factor):
    res = (x - g0) * np.exp(- (x - g0) **2 / (2 * sigma **2))
    remapped = x + factor * res
    return remapped.astype(x.dtype)

if __name__ == "__main__":
    x = np.arange(0, 10, 0.01)
    fig = plt.figure(figsize=(15, 4))
    # 测试tone mapping的remapping函数
    fig.add_subplot(131)
    plt.plot(x, x, 'g--', label='y=x')
    tone_out1 = remapping_tone(x, g0=5, sigma=1, beta=0.3)
    tone_out2 = remapping_tone(x, g0=5, sigma=1, beta=0.8)
    plt.plot(x, tone_out1, label='beta=0.3')
    plt.plot(x, tone_out2, label='beta=0.8')
    plt.grid()
    plt.legend()
    plt.title('Tone Mapping')
    # 测试smooth/enhance的remapping函数
    fig.add_subplot(132)
    plt.plot(x, x, 'g--', label='y=x')
    smooth_out1 = remapping_detail(x, g0=5, sigma=1, factor=-0.5)
    smooth_out2 = remapping_detail(x, g0=5, sigma=1, factor=-0.9)
    plt.plot(x, smooth_out1, label='factor=-0.3')
    plt.plot(x, smooth_out2, label='factor=-0.9')
    plt.grid()
    plt.legend()
    plt.title('Smooth')
    fig.add_subplot(133)
    plt.plot(x, x, 'g--', label='y=x')
    enhance_out1 = remapping_detail(x, g0=5, sigma=1, factor=0.8)
    enhance_out2 = remapping_detail(x, g0=5, sigma=1, factor=1.2)
    plt.plot(x, enhance_out1, label='factor=0.8')
    plt.plot(x, enhance_out2, label='factor=1.2')
    plt.grid()
    plt.legend()
    plt.title('Enhance')
    # 结果保存
    plt.savefig('results/llf/remap.png')

