from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


EPS = 1e-8
SQRT2 = np.sqrt(2.0)


@dataclass
class SVDResult:
    u: np.ndarray
    s: np.ndarray
    vt: np.ndarray


@dataclass
class DWTResult:
    coeffs: np.ndarray
    original_shape: tuple[int, int]
    padded_shape: tuple[int, int]
    levels: int


def load_image(image_path: Path) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    return np.asarray(image, dtype=np.float64)


def save_image(image: np.ndarray, output_path: Path) -> None:
    clipped = np.clip(np.rint(image), 0, 255).astype(np.uint8)
    Image.fromarray(clipped).save(output_path)


def normalize_vector(vec: np.ndarray) -> tuple[np.ndarray, float]:
    norm = float(np.linalg.norm(vec))
    if norm < EPS:
        return vec * 0.0, 0.0
    return vec / norm, norm


def manual_svd(matrix: np.ndarray, rank: int, max_iter: int = 150, tol: float = 1e-6) -> SVDResult:
    rows, cols = matrix.shape
    rank = max(1, min(rank, rows, cols))

    residual = matrix.astype(np.float64).copy()
    u_list: list[np.ndarray] = []
    singular_values: list[float] = []
    vt_list: list[np.ndarray] = []
    rng = np.random.default_rng(26)

    for _ in range(rank):
        v = rng.standard_normal(cols)
        v, v_norm = normalize_vector(v)
        if v_norm == 0.0:
            break

        sigma = 0.0
        for _ in range(max_iter):
            u, u_norm = normalize_vector(residual @ v)
            if u_norm == 0.0:
                sigma = 0.0
                break

            v_next, v_norm = normalize_vector(residual.T @ u)
            if v_norm == 0.0:
                sigma = 0.0
                break

            sigma_next = float(u @ residual @ v_next)
            if abs(sigma_next - sigma) <= tol * max(1.0, abs(sigma_next)):
                v = v_next
                sigma = sigma_next
                break

            v = v_next
            sigma = sigma_next

        if sigma <= tol:
            break

        u, _ = normalize_vector(residual @ v)
        sigma = float(u @ residual @ v)
        if sigma <= tol:
            break

        residual -= sigma * np.outer(u, v)
        u_list.append(u)
        singular_values.append(sigma)
        vt_list.append(v)

    if not singular_values:
        return SVDResult(
            u=np.zeros((rows, 0), dtype=np.float64),
            s=np.zeros((0,), dtype=np.float64),
            vt=np.zeros((0, cols), dtype=np.float64),
        )

    return SVDResult(
        u=np.column_stack(u_list),
        s=np.asarray(singular_values, dtype=np.float64),
        vt=np.vstack(vt_list),
    )


def reconstruct_from_svd(svd_result: SVDResult, rank: int) -> np.ndarray:
    use_rank = min(rank, svd_result.s.shape[0])
    if use_rank == 0:
        return np.zeros((svd_result.u.shape[0], svd_result.vt.shape[1]), dtype=np.float64)

    u = svd_result.u[:, :use_rank]
    s = svd_result.s[:use_rank]
    vt = svd_result.vt[:use_rank, :]
    return (u * s) @ vt


def precompute_image_svd(image: np.ndarray, max_rank: int) -> list[SVDResult]:
    return [manual_svd(image[:, :, idx], rank=max_rank) for idx in range(image.shape[2])]


def reconstruct_image_from_svd(svd_results: list[SVDResult], rank: int) -> np.ndarray:
    return np.stack([reconstruct_from_svd(result, rank) for result in svd_results], axis=2)


def compute_psnr_from_svd(
    image_shape: tuple[int, int, int],
    svd_results: list[SVDResult],
    rank: int,
) -> float:
    height, width, channels = image_shape
    total_squared_error = 0.0

    for result in svd_results:
        if rank < result.s.shape[0]:
            tail = result.s[rank:]
            total_squared_error += float(np.sum(tail * tail))

    mse = total_squared_error / float(height * width * channels)
    if mse < EPS:
        return float("inf")
    return 20.0 * np.log10(255.0 / np.sqrt(mse))


def compute_psnr(reference: np.ndarray, target: np.ndarray) -> float:
    mse = float(np.mean((reference - target) ** 2))
    if mse < EPS:
        return float("inf")
    return 20.0 * np.log10(255.0 / np.sqrt(mse))


def estimate_svd_storage_ratio(image_shape: tuple[int, int, int], rank: int) -> float:
    height, width, channels = image_shape
    original = height * width * channels
    compressed = channels * rank * (height + width + 1)
    return compressed / original


def estimate_svd_parameter_count(image_shape: tuple[int, int, int], rank: int) -> int:
    height, width, channels = image_shape
    return channels * rank * (height + width + 1)


def pad_to_multiple(channel: np.ndarray, multiple: int) -> tuple[np.ndarray, tuple[int, int]]:
    height, width = channel.shape
    target_h = ((height + multiple - 1) // multiple) * multiple
    target_w = ((width + multiple - 1) // multiple) * multiple
    pad_h = target_h - height
    pad_w = target_w - width
    padded = np.pad(channel, ((0, pad_h), (0, pad_w)), mode="edge")
    return padded, (height, width)


def haar_dwt2_once(block: np.ndarray) -> np.ndarray:
    low_rows = (block[:, 0::2] + block[:, 1::2]) / SQRT2
    high_rows = (block[:, 0::2] - block[:, 1::2]) / SQRT2
    temp = np.concatenate([low_rows, high_rows], axis=1)

    low_cols = (temp[0::2, :] + temp[1::2, :]) / SQRT2
    high_cols = (temp[0::2, :] - temp[1::2, :]) / SQRT2
    return np.concatenate([low_cols, high_cols], axis=0)


def haar_idwt2_once(block: np.ndarray) -> np.ndarray:
    half_h = block.shape[0] // 2
    half_w = block.shape[1] // 2

    temp = np.empty_like(block)
    temp[0::2, :] = (block[:half_h, :] + block[half_h:, :]) / SQRT2
    temp[1::2, :] = (block[:half_h, :] - block[half_h:, :]) / SQRT2

    restored = np.empty_like(block)
    restored[:, 0::2] = (temp[:, :half_w] + temp[:, half_w:]) / SQRT2
    restored[:, 1::2] = (temp[:, :half_w] - temp[:, half_w:]) / SQRT2
    return restored


def max_dwt_levels(height: int, width: int, requested_levels: int) -> int:
    max_possible = int(np.floor(np.log2(max(1, min(height, width)))))
    return max(1, min(requested_levels, max_possible))


def dwt_decompose(channel: np.ndarray, levels: int = 3) -> DWTResult:
    levels = max_dwt_levels(channel.shape[0], channel.shape[1], levels)
    padded, original_shape = pad_to_multiple(channel.astype(np.float64), 2 ** levels)
    coeffs = padded.copy()

    current_h, current_w = coeffs.shape
    for _ in range(levels):
        coeffs[:current_h, :current_w] = haar_dwt2_once(coeffs[:current_h, :current_w])
        current_h //= 2
        current_w //= 2

    return DWTResult(
        coeffs=coeffs,
        original_shape=original_shape,
        padded_shape=coeffs.shape,
        levels=levels,
    )


def dwt_reconstruct(result: DWTResult) -> np.ndarray:
    restored = result.coeffs.copy()
    current_h = result.padded_shape[0] // (2 ** (result.levels - 1))
    current_w = result.padded_shape[1] // (2 ** (result.levels - 1))

    for _ in range(result.levels):
        restored[:current_h, :current_w] = haar_idwt2_once(restored[:current_h, :current_w])
        current_h *= 2
        current_w *= 2

    orig_h, orig_w = result.original_shape
    return restored[:orig_h, :orig_w]


def compress_dwt_channel(channel: np.ndarray, keep_count: int, levels: int = 3) -> np.ndarray:
    transformed = dwt_decompose(channel, levels=levels)
    coeffs = transformed.coeffs
    total = coeffs.size
    keep_count = max(1, min(keep_count, total))

    flat_abs = np.abs(coeffs).reshape(-1)
    threshold_index = total - keep_count
    if threshold_index > 0:
        threshold = np.partition(flat_abs, threshold_index)[threshold_index]
        mask = np.abs(coeffs) >= threshold
        if int(mask.sum()) > keep_count:
            chosen = np.argpartition(flat_abs, threshold_index)[threshold_index:]
            exact_mask = np.zeros(total, dtype=bool)
            exact_mask[chosen] = True
            mask = exact_mask.reshape(coeffs.shape)
        transformed.coeffs = coeffs * mask

    return dwt_reconstruct(transformed)


def compress_image_dwt(
    image: np.ndarray,
    keep_count: int,
    levels: int = 3,
) -> np.ndarray:
    channels = image.shape[2]
    keep_per_channel = max(1, keep_count // channels)
    compressed_channels = [
        compress_dwt_channel(image[:, :, idx], keep_per_channel, levels=levels)
        for idx in range(channels)
    ]
    return np.stack(compressed_channels, axis=2)


def build_rank_grid(max_rank: int) -> list[int]:
    candidates = {1, 2, 4, 8, 12, 16, 24, 32, 48, 64, max_rank}
    return sorted(value for value in candidates if 1 <= value <= max_rank)


def print_svd_metrics(image: np.ndarray, rank_grid: list[int]) -> None:
    print("SVD")
    print("rank\tPSNR(dB)\tstorage_ratio")
    for rank in rank_grid:
        svd_results = precompute_image_svd(image, rank)
        compressed = reconstruct_image_from_svd(svd_results, rank)
        psnr = compute_psnr(image, compressed)
        ratio = estimate_svd_storage_ratio(image.shape, rank)
        print(f"{rank}\t{psnr:8.3f}\t{ratio:0.4f}")


def print_svd_dwt_comparison(image: np.ndarray, rank_grid: list[int], dwt_levels: int) -> None:
    print("rank\tSVD_PSNR\tDWT_PSNR\tstorage_ratio")
    for rank in rank_grid:
        svd_results = precompute_image_svd(image, rank)
        svd_image = reconstruct_image_from_svd(svd_results, rank)
        svd_psnr = compute_psnr(image, svd_image)
        storage_ratio = estimate_svd_storage_ratio(image.shape, rank)

        keep_count = min(
            int(round(storage_ratio * image.size)),
            image.size,
        )
        dwt_image = compress_image_dwt(image, keep_count=keep_count, levels=dwt_levels)
        dwt_psnr = compute_psnr(image, dwt_image)
        print(f"{rank}\t{svd_psnr:8.3f}\t{dwt_psnr:8.3f}\t{storage_ratio:0.4f}")


def launch_gui(image: np.ndarray, image_path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, Slider

    max_rank = min(image.shape[0], image.shape[1], 48)
    initial_rank = min(16, max_rank)
    svd_results = precompute_image_svd(image, max_rank)
    compressed = reconstruct_image_from_svd(svd_results, initial_rank)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 6))
    fig.subplots_adjust(bottom=0.23)

    ax_left.imshow(np.clip(image / 255.0, 0.0, 1.0))
    ax_left.set_title("Original")
    ax_left.axis("off")

    image_artist = ax_right.imshow(np.clip(compressed / 255.0, 0.0, 1.0))
    ax_right.axis("off")

    slider_ax = fig.add_axes((0.15, 0.12, 0.5, 0.03))
    slider = Slider(
        slider_ax,
        "Rank",
        1,
        max_rank,
        valinit=initial_rank,
        valstep=1,
        dragging=False,
    )

    button_ax = fig.add_axes((0.72, 0.08, 0.14, 0.06))
    save_button = Button(button_ax, "Save Result", color="lightblue", hovercolor="deepskyblue")

    status_text = fig.text(0.15, 0.05, "", fontsize=10)

    def refresh(rank: int) -> np.ndarray:
        current = reconstruct_image_from_svd(svd_results, rank)
        psnr = compute_psnr_from_svd(image.shape, svd_results, rank)
        ratio = estimate_svd_storage_ratio(image.shape, rank)
        image_artist.set_data(np.clip(current / 255.0, 0.0, 1.0))
        ax_right.set_title(f"SVD Compressed (rank={rank})")
        status_text.set_text(f"PSNR: {psnr:.3f} dB    storage ratio: {ratio:.4f}")
        fig.canvas.draw_idle()
        return current

    state = {"current": refresh(initial_rank)}

    def on_slider_change(value: float) -> None:
        state["current"] = refresh(int(value))

    def on_save(_: object) -> None:
        output_path = image_path.with_name(f"{image_path.stem}_svd_rank_{int(slider.val)}.png")
        save_image(state["current"], output_path)
        status_text.set_text(f"Saved to: {output_path}")
        fig.canvas.draw_idle()

    slider.on_changed(on_slider_change)
    save_button.on_clicked(on_save)
    plt.show()


def parse_args() -> argparse.Namespace:
    default_image = Path(__file__).resolve().parent.parent / "imgs" / "CMakeTools.png"
    parser = argparse.ArgumentParser(description="Option 1: manual SVD image compression")
    parser.add_argument("--image", type=Path, default=default_image, help="input image path")
    parser.add_argument("--rank", type=int, default=24, help="rank for one-shot compression")
    parser.add_argument("--method", choices=("svd", "dwt"), default="svd", help="compression method")
    parser.add_argument("--dwt-levels", type=int, default=3, help="levels for Haar DWT compression")
    parser.add_argument("--compare-dwt", action="store_true", help="print SVD vs DWT comparison table")
    parser.add_argument("--no-gui", action="store_true", help="skip the GUI and only save one result")
    parser.add_argument("--output", type=Path, default=None, help="output image path for --no-gui mode")
    parser.add_argument("--print-table", action="store_true", help="print metric table")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = args.image.resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = load_image(image_path)
    max_rank = min(image.shape[0], image.shape[1])
    rank = max(1, min(args.rank, max_rank))

    if args.print_table:
        rank_grid = build_rank_grid(min(max_rank, 64))
        if args.compare_dwt:
            print_svd_dwt_comparison(image, rank_grid, args.dwt_levels)
        else:
            print_svd_metrics(image, rank_grid)

    if args.no_gui:
        if args.method == "svd":
            svd_results = precompute_image_svd(image, rank)
            compressed = reconstruct_image_from_svd(svd_results, rank)
            psnr = compute_psnr(image, compressed)
            ratio = estimate_svd_storage_ratio(image.shape, rank)
            output_name = f"{image_path.stem}_svd_rank_{rank}.png"
        else:
            keep_count = min(estimate_svd_parameter_count(image.shape, rank), image.size)
            compressed = compress_image_dwt(image, keep_count=keep_count, levels=args.dwt_levels)
            psnr = compute_psnr(image, compressed)
            ratio = keep_count / float(image.size)
            output_name = f"{image_path.stem}_dwt_rank_{rank}.png"

        output_path = args.output or image_path.with_name(output_name)
        save_image(compressed, output_path)
        print(f"Saved result to: {output_path}")
        print(f"method={args.method}, rank={rank}, PSNR={psnr:.3f} dB, storage ratio={ratio:.4f}")
        return

    if args.method != "svd":
        raise ValueError("GUI mode currently supports SVD only. Use --no-gui for DWT.")

    launch_gui(image, image_path)


if __name__ == "__main__":
    main()
