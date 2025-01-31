import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift
from PIL import Image

# 读取图片并转换为灰度图像
import numpy as np
from PIL import Image
from numpy.fft import fft2, fftshift

def load_image(path):
    image = np.load(path)
    image = image.squeeze(0).transpose(1, 2, 0)
    gray_image = 0.2989 * image[:, :, 0] + 0.5870 * image[:, :, 1] + 0.1140 * image[:, :, 2]
    return gray_image

# # 读取.npy文件
# data = np.load(file_path)
def calculate_frequency_energy_ratio(image_array):
    # 应用二维傅里叶变换
    frequency_transform = fft2(image_array)
    frequency_shifted = fftshift(frequency_transform)
    magnitude_spectrum = np.abs(frequency_shifted)
    
    # 计算总能量
    total_energy = np.mean(magnitude_spectrum)
    
    # 计算低频和高频能量
    # 假设低频在中心的1/4区域
    center = np.array(magnitude_spectrum.shape) // 2
    low_freq_width = magnitude_spectrum.shape[0] // 4
    low_freq_area = magnitude_spectrum[
        center[0] - low_freq_width//2:center[0] + low_freq_width//2,
        center[1] - low_freq_width//2:center[1] + low_freq_width//2
    ]
    # low_freq_area = magnitude_spectrum[
    #     center[0] - low_freq_width:center[0] + low_freq_width,
    #     center[1] - low_freq_width:center[1] + low_freq_width
    # ]
    low_freq_energy = np.mean(low_freq_area)
    high_freq_energy = (16*total_energy - low_freq_energy)/15
    # 计算比例
    energy_ratio = high_freq_energy/low_freq_energy
    return energy_ratio

ratio={}
for datasize in [100,50,30,15,10]:
    ratio[datasize]=[]
    for i in range(999, -1, -1):
        img_path = '/home/zsj/code/dps/rebuttal/gd_freq_{}/{}.npy'.format(datasize,i)  # 替换为你的图片路径
        image_data = load_image(img_path)
        frequency_energy_ratio = calculate_frequency_energy_ratio(image_data)
        # print(img_path,frequency_energy_ratio)
        # print(sdsd)
        ratio[datasize].append(frequency_energy_ratio)
ratio_arrays = np.stack(list(ratio.values()))
print(ratio_arrays.shape)
ratio_arrays_gd = np.mean(ratio_arrays, axis=0)

ratio={}
for datasize in [100,50,30,15,10]:
    ratio[datasize]=[]
    for i in range(999, -1, -1):
        img_path = '/home/zsj/code/dps/rebuttal/sr_freq_{}/{}.npy'.format(datasize,i)  # 替换为你的图片路径
        image_data = load_image(img_path)
        frequency_energy_ratio = calculate_frequency_energy_ratio(image_data)
        ratio[datasize].append(frequency_energy_ratio)
ratio_arrays = np.stack(list(ratio.values()))
ratio_arrays_sr = np.mean(ratio_arrays, axis=0)

ratio={}
for datasize in [100,50,30,15,10]:
    ratio[datasize]=[]
    for i in range(999, -1, -1):
        img_path = '/home/zsj/code/dps/rebuttal/ip_freq_{}/{}.npy'.format(datasize,i)  # 替换为你的图片路径
        image_data = load_image(img_path)
        frequency_energy_ratio = calculate_frequency_energy_ratio(image_data)
        ratio[datasize].append(frequency_energy_ratio)
ratio_arrays = np.stack(list(ratio.values()))
ratio_arrays_ip = np.mean(ratio_arrays, axis=0)

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import style

# # Simulating 99 data points
# # x = np.linspace(0, 999, 1000)
# x = np.arange(999, 0, -1)
# # y = ratio
# plt.xticks(np.arange(1000, -1, -100))

# # Applying NeurIPS style
# style.use('ggplot')

# # Creating the plot
# plt.figure(figsize=(8, 6))
# plt.plot(x, ratio_arrays, label='ip')
# # plt.plot(x, average_ratio_gd, label='gd')
# # plt.plot(x, average_ratio_sr, label='sr')

# plt.title('NeurIPS Style Plot')
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.legend()
# plt.grid(True)

# # Saving the plot as a PDF
# plt.savefig('NeurIPS_Style_Plot.pdf')

# # Showing the plot
# plt.show()



import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
# 假设 data 是您的 1000 个数据点组成的数组
# 创建一个从 1000 到 1 的递减的 x 轴坐标数组
x = np.arange(1000, 0, -1)
style.use('ggplot')

# 绘图
plt.figure(figsize=(8, 6))  # 可以调整图形的大小
plt.plot(x, ratio_arrays_gd,label='Gaussian Deblur')
plt.plot(x, ratio_arrays_sr,label='Super Resolution')
plt.plot(x, ratio_arrays_ip,label='Inpainting(random)')
plt.xlabel('DDPM Time Steps t',fontsize=13)
plt.ylabel('Ratio',fontsize=13)
plt.title('Mean Magnitude Ratio (High Frequency/Low Frequency)',fontsize=13)
plt.grid(True)
plt.legend(fontsize=13)
plt.grid(True)

# 设置 x 轴的显示范围从 1000 到 1
plt.xlim(1000, 0)
plt.tight_layout()
plt.savefig('freq_ratio_2.pdf',dpi=600,bbox_inches='tight', pad_inches=0)
plt.show()

