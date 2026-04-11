import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from skimage import io

## read image
im = io.imread('../figs/original.png')
if im.ndim == 3 and im.shape[2] == 4:
    im = im[:, :, :3]

## draw 2 copies of the image
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
fig.subplots_adjust(bottom=0.22)
ax1.imshow(im)
ax1.set_title('Input image')
ax1.axis('off')
himg = ax2.imshow(np.zeros_like(im))
ax2.set_title('Resized Image\nAdjust sliders and click the button')
ax2.axis('off')

slider_col_ax = fig.add_axes((0.15, 0.10, 0.30, 0.03))
slider_row_ax = fig.add_axes((0.15, 0.05, 0.30, 0.03))
slider_col = Slider(slider_col_ax, 'Col scale', 0.5, 2.0, valinit=1.0)
slider_row = Slider(slider_row_ax, 'Row scale', 0.5, 2.0, valinit=1.0)

btn_ax = fig.add_axes((0.60, 0.06, 0.20, 0.06))
btn = Button(btn_ax, 'Seam Carving', color='lightblue', hovercolor='deepskyblue')

def on_click(event):
    h, w = im.shape[:2]
    target_w = max(1, int(w * slider_col.val))
    target_h = max(1, int(h * slider_row.val))
    result = seam_carve_image(im, (target_h, target_w))
    himg.set_data(result)
    himg.set_extent([0, result.shape[1], result.shape[0], 0])
    ax2.set_title(f'Resized Image ({result.shape[0]}x{result.shape[1]})')
    fig.canvas.draw_idle()

btn.on_clicked(on_click)


LAPLACIAN_KERNEL = np.array([
    [0.5, 1.0, 0.5],
    [1.0, -6.0, 1.0],
    [0.5, 1.0, 0.5],
], dtype=np.float32)

"""
第一部分是能量函数。按照作业要求，采用了基于 Laplacian 的能量。具体做法是，对 RGB 三个通道
分别做一个 3 乘 3 的离散拉普拉斯卷积，再把三个通道响应的平方加起来，得到每个像素的能量值。
先给整个图像的四条边各自补上一行，用来做边缘的计算。由于对每个点计算能量的方法都是一样的，所
以可以直接把整张图像做一个偏移，然后乘卷积核对应的元素，再把结果加起来求平方，得到整个图像上
每一点的能量。
"""

def compute_energy(im):
    im = im.astype(np.float32, copy=False)
    h, w, ch = im.shape
    energy = np.zeros((h, w), dtype=np.float32)

    for c in range(ch):
        channel = np.pad(im[:, :, c], ((1, 1), (1, 1)), mode='edge')

        response = np.zeros((h, w), dtype=np.float32)
        for dy in range(3):
            for dx in range(3):
                response += (
                    LAPLACIAN_KERNEL[dy, dx] *
                    channel[dy:dy+h, dx:dx+w]
                )

        energy += response * response

    return energy

"""
第二部分是最优 seam 的搜索。对于一条竖直 seam，它要求每一行只经过一个像素，
并且相邻两行的位置偏移不超过 1。给出两个矩阵：cost和parent，分别代表到达该
点需要的最小能量，以及对应路径的上一层坐标。为了找到总能量最小的一条路径，我
用了动态规划。从第一行往下规划，到达第一行的最小能量，自然是它本身的能量；到
达第二行的最小能量，只需要考虑它的左上、正上、右上三个方向的能量，哪个最小就
可以了，把那个最小的能量加上第二行对应像素的能量，再把最小能量的位置写入parent矩阵
对应位置。以此类推直到最后一行。
接下来，取最后一行累计能量最小的像素，根据parent矩阵一直回溯到第一行，返回这
条路径的各点坐标。
"""

def find_vertical_seam(energy):
    """Find the minimum-energy vertical seam with dynamic programming."""
    h, w = energy.shape
    cost = energy.copy()
    parent = np.zeros((h, w), dtype=np.int32)

    for i in range(1, h):
        prev_row = cost[i - 1]
        best_cost = prev_row.copy()
        best_parent = np.arange(w, dtype=np.int32)

        left_better = prev_row[:-1] < best_cost[1:]
        best_cost[1:][left_better] = prev_row[:-1][left_better]
        best_parent[1:][left_better] -= 1

        right_better = prev_row[1:] < best_cost[:-1]
        best_cost[:-1][right_better] = prev_row[1:][right_better]
        best_parent[:-1][right_better] += 1

        parent[i] = best_parent
        cost[i] += best_cost

    seam = np.zeros(h, dtype=np.int32)
    seam[-1] = np.argmin(cost[-1])
    for i in range(h - 2, -1, -1):
        seam[i] = parent[i + 1, seam[i + 1]]

    return seam

"""
第三部分是 seam 的实际操作。缩小图像时，创建一个mask布尔矩阵，初始值全是True，
代表每个位置像素是否保留。把每个需要删去的位置改成False，然后按照mask矩阵，把图
像写入新的少一列的图像。
"""

def remove_vertical_seam(im, seam):
    """Remove one vertical seam from the image."""
    h, w, c = im.shape
    mask = np.ones((h, w), dtype=bool)
    mask[np.arange(h), seam] = False
    return im[mask].reshape(h, w - 1, c)

"""
不考虑颜色通道的删除函数，用来保存删去像素的索引。
"""

def remove_vertical_seam_2d(arr, seam):
    """Remove one vertical seam from a 2D array."""
    h, w = arr.shape
    mask = np.ones((h, w), dtype=bool)
    mask[np.arange(h), seam] = False
    return arr[mask].reshape(h, w - 1)

"""
找出前n条能量最小的竖直路径。先找最小的，删掉；再找最小的，删掉……
"""

def collect_vertical_seams(im, k):
    """Find the first k removable seams and map them back to original coordinates."""
    temp_im = im.copy()
    h, w, _ = temp_im.shape
    index_map = np.tile(np.arange(w, dtype=np.int32), (h, 1))
    seams = []

    for _ in range(k):
        seam = find_vertical_seam(compute_energy(temp_im))
        seams.append(index_map[np.arange(h), seam].copy())
        temp_im = remove_vertical_seam(temp_im, seam)
        index_map = remove_vertical_seam_2d(index_map, seam)

    return seams

"""
插入新的竖直列，只需要取能量最小的竖直路径，在它的右侧插入新的一列像素，
新像素的值取决于它左右两侧像素平均值。如果本身就在右侧边界上，继承左侧像素。
列比较多时，依次插入到能量最小的那些列右侧。
"""

def insert_vertical_seams(im, seams):
    """Insert multiple seams using the original-image coordinates."""
    if not seams:
        return im

    h, w, c = im.shape
    seams = np.stack(seams, axis=0)
    out = np.empty((h, w + seams.shape[0], c), dtype=im.dtype)

    for i in range(h):
        row_positions = np.sort(seams[:, i])
        dst = 0
        seam_ptr = 0

        for j in range(w):
            out[i, dst, :] = im[i, j, :]
            dst += 1

            while seam_ptr < row_positions.size and row_positions[seam_ptr] == j:
                neighbor = min(j + 1, w - 1)
                out[i, dst, :] = 0.5 * (im[i, j, :] + im[i, neighbor, :])
                dst += 1
                seam_ptr += 1

    return out

"""
根据缩放倍数，求出需要增加或减少的列数，然后进行消除或者插入
"""

def enlarge_width(im, target_w, max_step_ratio=1.5):
    """Enlarge image width by inserting precomputed seams in bounded steps."""
    out = im
    while out.shape[1] < target_w:
        step_target = min(target_w, max(out.shape[1] + 1, int(np.ceil(out.shape[1] * max_step_ratio))))
        k = step_target - out.shape[1]
        seams = collect_vertical_seams(out, k)
        out = insert_vertical_seams(out, seams)
    return out


def seam_carve_image(im, sz):
    """Seam carving to resize image to target size.

    Args:
        im: (h, w, 3) input RGB image (uint8)
        sz: (target_h, target_w) target size

    Returns:
        resized image of shape (target_h, target_w, 3)
    """
    target_h, target_w = sz
    out = im.astype(np.float32).copy()

    if out.ndim != 3 or out.shape[2] != 3:
        raise ValueError('Expected an RGB image with shape (h, w, 3).')
    if target_h < 1 or target_w < 1:
        raise ValueError('Target size must be positive.')

    while out.shape[1] > target_w:
        seam = find_vertical_seam(compute_energy(out))
        out = remove_vertical_seam(out, seam)

    while out.shape[0] > target_h:
        out = np.transpose(out, (1, 0, 2))
        seam = find_vertical_seam(compute_energy(out))
        out = remove_vertical_seam(out, seam)
        out = np.transpose(out, (1, 0, 2))

    if out.shape[1] < target_w:
        out = enlarge_width(out, target_w)

    if out.shape[0] < target_h:
        out = np.transpose(out, (1, 0, 2))
        out = enlarge_width(out, target_h)
        out = np.transpose(out, (1, 0, 2))

    return np.clip(out, 0, 255).astype(np.uint8)


plt.show()
