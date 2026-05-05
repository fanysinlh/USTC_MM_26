from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, cast

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline, PchipInterpolator, RBFInterpolator, splprep, splev
from scipy.spatial import cKDTree # type: ignore


Array = NDArray[np.float64]
EPS = 1e-12


@dataclass
class CurveData:
    name: str
    points: Array
    closed: bool
    truth: Array | None = None


@dataclass
class FitResult:
    method: str
    parameterization: str
    dense_curve: Array
    fitted_at_samples: Array
    sample_rmse: float
    sample_to_curve_rmse: float
    truth_to_curve_rmse: float | None
    curve_to_truth_rmse: float | None
    chamfer: float | None


def parameterize_points(points: Array, mode: str, closed: bool = False) -> Array:
    points = np.asarray(points, dtype=np.float64)
    if points.shape[0] < 2:
        return np.array([0.0], dtype=np.float64)

    if closed:
        deltas = np.linalg.norm(np.diff(np.vstack([points, points[0]]), axis=0), axis=1)
    else:
        deltas = np.linalg.norm(np.diff(points, axis=0), axis=1)

    if mode == "uniform":
        steps = np.ones(deltas.shape[0], dtype=np.float64)
    elif mode == "chord":
        steps = np.maximum(deltas, EPS)
    elif mode == "centripetal":
        steps = np.sqrt(np.maximum(deltas, EPS))
    else:
        raise ValueError(f"Unknown parameterization: {mode}")

    if closed:
        params = np.concatenate(([0.0], np.cumsum(steps[:-1])))
        total = float(np.sum(steps))
    else:
        params = np.concatenate(([0.0], np.cumsum(steps)))
        total = float(params[-1])

    if total <= EPS:
        return np.linspace(0.0, 1.0, points.shape[0], dtype=np.float64)
    return params / total


def evaluate_chamfer(curve_a: Array, curve_b: Array) -> float:
    tree_a = cKDTree(curve_a)
    tree_b = cKDTree(curve_b)
    dist_ab = np.asarray(tree_b.query(curve_a)[0], dtype=np.float64)
    dist_ba = np.asarray(tree_a.query(curve_b)[0], dtype=np.float64)
    return float(0.5 * (np.mean(dist_ab**2) + np.mean(dist_ba**2)))


def directed_rmse(source: Array, target: Array) -> float:
    tree = cKDTree(target)
    distances = np.asarray(tree.query(source)[0], dtype=np.float64)
    return float(np.sqrt(np.mean(distances**2)))


def fit_cubic_interpolant(points: Array, params: Array, closed: bool, samples: int = 400) -> tuple[Array, Array]:
    bc_type = "periodic" if closed else "natural"
    periodic_points = points
    periodic_params = params

    if closed:
        periodic_points = np.vstack([points, points[0]])
        periodic_params = np.concatenate([params, [1.0]])

    spline_x = CubicSpline(periodic_params, periodic_points[:, 0], bc_type=bc_type)
    spline_y = CubicSpline(periodic_params, periodic_points[:, 1], bc_type=bc_type)

    dense_t = np.linspace(0.0, 1.0, samples, endpoint=not closed)
    dense_curve = np.column_stack([spline_x(dense_t), spline_y(dense_t)])
    fitted_at_samples = np.column_stack([spline_x(params), spline_y(params)])
    return dense_curve, fitted_at_samples


def fit_pchip_interpolant(points: Array, params: Array, closed: bool, samples: int = 400) -> tuple[Array, Array]:
    fitted_points = points
    fitted_params = params
    if closed:
        fitted_points = np.vstack([points, points[0]])
        fitted_params = np.concatenate([params, [1.0]])

    interpolator_x = PchipInterpolator(fitted_params, fitted_points[:, 0])
    interpolator_y = PchipInterpolator(fitted_params, fitted_points[:, 1])
    dense_t = np.linspace(0.0, 1.0, samples, endpoint=not closed)
    dense_curve = np.column_stack([interpolator_x(dense_t), interpolator_y(dense_t)])
    fitted_at_samples = np.column_stack([interpolator_x(params), interpolator_y(params)])
    return dense_curve, fitted_at_samples


def fit_rbf_interpolant(points: Array, params: Array, closed: bool, samples: int = 400) -> tuple[Array, Array]:
    fitted_points = points
    fitted_params = params
    if closed:
        fitted_points = np.vstack([points, points[0]])
        fitted_params = np.concatenate([params, [1.0]])

    rbf = RBFInterpolator(fitted_params[:, None], fitted_points, kernel="thin_plate_spline")
    dense_t = np.linspace(0.0, 1.0, samples, endpoint=not closed)
    dense_curve = np.asarray(rbf(dense_t[:, None]), dtype=np.float64)
    fitted_at_samples = np.asarray(rbf(params[:, None]), dtype=np.float64)
    return dense_curve, fitted_at_samples


def fit_smoothing_spline(
    points: Array,
    params: Array,
    closed: bool,
    smooth_scale: float = 0.002,
    samples: int = 400,
) -> tuple[Array, Array]:
    curve_scale = max(float(np.linalg.norm(np.ptp(points, axis=0))), 1.0)
    smoothing = smooth_scale * points.shape[0] * (curve_scale**2)
    tck, _ = splprep(
        [points[:, 0], points[:, 1]],
        u=params,
        s=smoothing,
        per=int(closed),
        k=min(3, points.shape[0] - 1),
    )
    dense_t = np.linspace(0.0, 1.0, samples, endpoint=not closed)
    dense_eval = splev(dense_t, tck)
    sample_eval = splev(params, tck)
    dense_curve = np.column_stack(dense_eval)
    fitted_at_samples = np.column_stack(sample_eval)
    return dense_curve, fitted_at_samples


def run_fit(
    data: CurveData,
    method: str,
    parameterization: str,
    smooth_scale: float = 0.002,
    samples: int = 400,
) -> FitResult:
    params = parameterize_points(data.points, parameterization, closed=data.closed)
    if method == "interpolation":
        dense_curve, fitted_at_samples = fit_cubic_interpolant(data.points, params, data.closed, samples)
    elif method == "pchip":
        dense_curve, fitted_at_samples = fit_pchip_interpolant(data.points, params, data.closed, samples)
    elif method == "rbf":
        dense_curve, fitted_at_samples = fit_rbf_interpolant(data.points, params, data.closed, samples)
    elif method == "smoothing":
        dense_curve, fitted_at_samples = fit_smoothing_spline(
            data.points,
            params,
            data.closed,
            smooth_scale=smooth_scale,
            samples=samples,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    residual = fitted_at_samples - data.points
    sample_rmse = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
    sample_to_curve_rmse = directed_rmse(data.points, dense_curve)
    truth_to_curve_rmse = directed_rmse(data.truth, dense_curve) if data.truth is not None else None
    curve_to_truth_rmse = directed_rmse(dense_curve, data.truth) if data.truth is not None else None
    chamfer = evaluate_chamfer(dense_curve, data.truth) if data.truth is not None else None
    return FitResult(
        method=method,
        parameterization=parameterization,
        dense_curve=dense_curve,
        fitted_at_samples=fitted_at_samples,
        sample_rmse=sample_rmse,
        sample_to_curve_rmse=sample_to_curve_rmse,
        truth_to_curve_rmse=truth_to_curve_rmse,
        curve_to_truth_rmse=curve_to_truth_rmse,
        chamfer=chamfer,
    )


def flower_curve(t: Array) -> Array:
    theta = 2.0 * np.pi * t
    radius = 1.0 + 0.28 * np.cos(5.0 * theta)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return np.column_stack([x, y])


def open_wave_curve(t: Array) -> Array:
    x = 2.2 * t - 1.1
    y = 0.45 * np.sin(2.5 * np.pi * t) + 0.15 * np.sin(7.0 * np.pi * t)
    return np.column_stack([x, y])


def sample_curve(
    curve_fn: Callable[[Array], Array],
    sample_count: int,
    closed: bool,
    noise_std: float,
    seed: int,
) -> CurveData:
    rng = np.random.default_rng(seed)
    sample_t = np.linspace(0.0, 1.0, sample_count, endpoint=not closed)
    truth_t = np.linspace(0.0, 1.0, 2000, endpoint=not closed)
    truth = curve_fn(truth_t)
    samples = curve_fn(sample_t)
    if noise_std > 0.0:
        samples = samples + rng.normal(scale=noise_std, size=samples.shape)
    return CurveData(
        name=curve_fn.__name__,
        points=samples,
        closed=closed,
        truth=truth,
    )


def sample_curve_at_parameters(
    curve_fn: Callable[[Array], Array],
    sample_t: Array,
    closed: bool,
    noise_std: float,
    seed: int,
) -> CurveData:
    rng = np.random.default_rng(seed)
    truth_t = np.linspace(0.0, 1.0, 2000, endpoint=not closed)
    truth = curve_fn(truth_t)
    samples = curve_fn(sample_t)
    if noise_std > 0.0:
        samples = samples + rng.normal(scale=noise_std, size=samples.shape)
    return CurveData(
        name=curve_fn.__name__,
        points=samples,
        closed=closed,
        truth=truth,
    )


def load_points(path: Path) -> Array:
    raw = np.asarray(np.loadtxt(path, dtype=np.float64), dtype=np.float64)
    if raw.ndim == 1:
        raw = raw[None, :]
    if raw.shape[1] < 2:
        raise ValueError("Input point file must contain at least two columns")
    return raw[:, :2]


def require_truth(data: CurveData) -> Array:
    if data.truth is None:
        raise ValueError("This plotting routine requires ground-truth curve data")
    return data.truth


def save_metrics_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_metric(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.6f}"


def pretty_method_name(method: str) -> str:
    return {
        "interpolation": "Cubic spline",
        "pchip": "PCHIP",
        "rbf": "RBF",
        "smoothing": "Smoothing spline",
    }[method]


def plot_parameterization_comparison(data: CurveData, output_path: Path) -> list[dict[str, str]]:
    modes = ["uniform", "chord", "centripetal"]
    rows: list[dict[str, str]] = []
    truth = require_truth(data)
    fig, axes = plt.subplots(1, len(modes), figsize=(15, 4.8), constrained_layout=True)
    axes_array = np.atleast_1d(axes).astype(object)

    for ax_obj, mode in zip(axes_array, modes):
        ax = cast(Axes, ax_obj)
        result = run_fit(data, method="interpolation", parameterization=mode)
        ax.plot(truth[:, 0], truth[:, 1], color="#b8c0c7", linewidth=2.4, label="ground truth")
        ax.plot(result.dense_curve[:, 0], result.dense_curve[:, 1], color="#007f5f", linewidth=2.2, label="fitted curve")
        ax.scatter(data.points[:, 0], data.points[:, 1], color="#c1121f", s=28, zorder=3, label="samples")
        ax.set_title(
            f"{mode}\nRMSE={result.sample_rmse:.4f}, Chamfer={result.chamfer:.5f}",
            fontsize=11,
        )
        ax.set_aspect("equal")
        ax.grid(alpha=0.25)
        rows.append(
            {
                "experiment": "parameterization",
                "method": result.method,
                "parameterization": result.parameterization,
                "sample_rmse": format_metric(result.sample_rmse),
                "sample_to_curve_rmse": format_metric(result.sample_to_curve_rmse),
                "truth_to_curve_rmse": format_metric(result.truth_to_curve_rmse),
                "curve_to_truth_rmse": format_metric(result.curve_to_truth_rmse),
                "chamfer": format_metric(result.chamfer),
            }
        )

    first_ax = cast(Axes, axes_array[0])
    handles, labels = first_ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Cubic spline interpolation under different parameterizations", fontsize=15, y=1.02)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return rows


def plot_noise_comparison(data: CurveData, output_path: Path) -> list[dict[str, str]]:
    configs = [
        ("interpolation", "chord", 0.0, "Interpolation"),
        ("smoothing", "chord", 0.0005, "Moderate smoothing"),
        ("smoothing", "centripetal", 0.0001, "Centripetal smoothing"),
    ]
    rows: list[dict[str, str]] = []
    truth = require_truth(data)
    fig, axes = plt.subplots(1, len(configs), figsize=(15, 4.8), constrained_layout=True)
    axes_array = np.atleast_1d(axes).astype(object)

    for ax_obj, (method, mode, smooth_scale, title) in zip(axes_array, configs):
        ax = cast(Axes, ax_obj)
        result = run_fit(data, method=method, parameterization=mode, smooth_scale=smooth_scale)
        ax.plot(truth[:, 0], truth[:, 1], color="#b8c0c7", linewidth=2.4, label="ground truth")
        ax.plot(result.dense_curve[:, 0], result.dense_curve[:, 1], color="#00509d", linewidth=2.2, label="fitted curve")
        ax.scatter(data.points[:, 0], data.points[:, 1], color="#fb8500", s=28, zorder=3, label="noisy samples")
        ax.set_title(
            f"{title}\nRMSE={result.sample_rmse:.4f}, Chamfer={result.chamfer:.5f}",
            fontsize=11,
        )
        ax.set_aspect("equal")
        ax.grid(alpha=0.25)
        rows.append(
            {
                "experiment": "noise_robustness",
                "method": result.method,
                "parameterization": result.parameterization,
                "sample_rmse": format_metric(result.sample_rmse),
                "sample_to_curve_rmse": format_metric(result.sample_to_curve_rmse),
                "truth_to_curve_rmse": format_metric(result.truth_to_curve_rmse),
                "curve_to_truth_rmse": format_metric(result.curve_to_truth_rmse),
                "chamfer": format_metric(result.chamfer),
            }
        )

    first_ax = cast(Axes, axes_array[0])
    handles, labels = first_ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Interpolation vs fitting under noisy observations", fontsize=15, y=1.02)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return rows


def plot_custom_fit(
    points: Array,
    closed: bool,
    method: str,
    parameterization: str,
    smooth_scale: float,
    output_path: Path,
) -> None:
    data = CurveData(name="custom", points=points, closed=closed, truth=None)
    result = run_fit(data, method=method, parameterization=parameterization, smooth_scale=smooth_scale)
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    ax.scatter(points[:, 0], points[:, 1], color="#d62828", s=30, label="input points")
    ax.plot(result.dense_curve[:, 0], result.dense_curve[:, 1], color="#1d3557", linewidth=2.2, label="fitted curve")
    ax.set_title(
        f"Custom fit: {method}, {parameterization}\nSample RMSE={result.sample_rmse:.4f}",
        fontsize=12,
    )
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_method_comparison(data: CurveData, output_path: Path) -> list[dict[str, str]]:
    methods = ["interpolation", "pchip", "rbf", "smoothing"]
    truth = require_truth(data)
    rows: list[dict[str, str]] = []
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.8), constrained_layout=True)
    axes_array = np.atleast_1d(axes).reshape(-1).astype(object)

    for ax_obj, method in zip(axes_array, methods):
        ax = cast(Axes, ax_obj)
        smooth_scale = 0.0002 if method == "smoothing" else 0.0
        result = run_fit(data, method=method, parameterization="chord", smooth_scale=smooth_scale)
        ax.plot(truth[:, 0], truth[:, 1], color="#c7d0d9", linewidth=2.2, label="ground truth")
        ax.plot(result.dense_curve[:, 0], result.dense_curve[:, 1], color="#15616d", linewidth=2.1, label="fitted curve")
        ax.scatter(data.points[:, 0], data.points[:, 1], color="#ff7d00", s=25, zorder=3, label="samples")
        ax.set_title(
            f"{pretty_method_name(method)}\nSample RMSE={result.sample_rmse:.4f}, Chamfer={result.chamfer:.5f}",
            fontsize=11,
        )
        ax.set_aspect("equal")
        ax.grid(alpha=0.25)
        rows.append(
            {
                "experiment": "method_comparison",
                "method": method,
                "parameterization": "chord",
                "sample_rmse": format_metric(result.sample_rmse),
                "sample_to_curve_rmse": format_metric(result.sample_to_curve_rmse),
                "truth_to_curve_rmse": format_metric(result.truth_to_curve_rmse),
                "curve_to_truth_rmse": format_metric(result.curve_to_truth_rmse),
                "chamfer": format_metric(result.chamfer),
            }
        )

    first_ax = cast(Axes, axes_array[0])
    handles, labels = first_ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Comparison of different fitting/interpolation functions", fontsize=15, y=1.02)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return rows


def plot_noise_robustness_curve(output_path: Path) -> list[dict[str, str]]:
    noise_levels = np.array([0.0, 0.02, 0.04, 0.06, 0.08, 0.10], dtype=np.float64)
    methods = [
        ("interpolation", "Cubic spline", "#b02e0c", 0.0),
        ("pchip", "PCHIP", "#2a6f97", 0.0),
        ("rbf", "RBF", "#7b2cbf", 0.0),
        ("smoothing", "Smoothing spline", "#2b9348", 0.0005),
    ]

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    rows: list[dict[str, str]] = []
    for method, label, color, smooth_scale in methods:
        chamfer_values: list[float] = []
        for noise in noise_levels:
            data = sample_curve(open_wave_curve, sample_count=24, closed=False, noise_std=float(noise), seed=26)
            result = run_fit(data, method=method, parameterization="chord", smooth_scale=smooth_scale)
            chamfer_values.append(float(result.chamfer if result.chamfer is not None else np.nan))
            rows.append(
                {
                    "experiment": "noise_sweep",
                    "method": method,
                    "parameterization": "chord",
                    "noise_std": f"{noise:.2f}",
                    "sample_rmse": format_metric(result.sample_rmse),
                    "sample_to_curve_rmse": format_metric(result.sample_to_curve_rmse),
                    "truth_to_curve_rmse": format_metric(result.truth_to_curve_rmse),
                    "curve_to_truth_rmse": format_metric(result.curve_to_truth_rmse),
                    "chamfer": format_metric(result.chamfer),
                }
            )
        ax.plot(noise_levels, chamfer_values, marker="o", linewidth=2.0, color=color, label=label)

    ax.set_xlabel("Noise standard deviation")
    ax.set_ylabel("Symmetric Chamfer distance")
    ax.set_title("Robustness under increasing observation noise")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return rows


def build_metric_definition_rows(data: CurveData) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for method, parameterization, smooth_scale in [
        ("interpolation", "chord", 0.0),
        ("smoothing", "chord", 0.0005),
    ]:
        result = run_fit(data, method=method, parameterization=parameterization, smooth_scale=smooth_scale)
        rows.append(
            {
                "experiment": "metric_definition",
                "method": method,
                "parameterization": parameterization,
                "sample_rmse": format_metric(result.sample_rmse),
                "sample_to_curve_rmse": format_metric(result.sample_to_curve_rmse),
                "truth_to_curve_rmse": format_metric(result.truth_to_curve_rmse),
                "curve_to_truth_rmse": format_metric(result.curve_to_truth_rmse),
                "chamfer": format_metric(result.chamfer),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Option 1: parametric curve fitting for planar point sets")
    parser.add_argument(
        "--mode",
        choices=("demo", "custom"),
        default="demo",
        help="run built-in experiments or fit a custom point file",
    )
    parser.add_argument("--input", type=Path, default=None, help="path to a point file with at least two columns")
    parser.add_argument("--closed", action="store_true", help="treat custom input as a closed curve")
    parser.add_argument(
        "--method",
        choices=("interpolation", "pchip", "rbf", "smoothing"),
        default="smoothing",
        help="curve fitting method for custom mode",
    )
    parser.add_argument(
        "--parameterization",
        choices=("uniform", "chord", "centripetal"),
        default="chord",
        help="parameterization strategy for custom mode",
    )
    parser.add_argument("--smooth-scale", type=float, default=0.002, help="relative smoothing strength")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "report" / "figures")
    parser.add_argument("--list-methods", action="store_true", help="print supported fitting methods and exit")
    return parser.parse_args()


def run_demo(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    clustered_t = np.array(
        [
            0.00,
            0.03,
            0.06,
            0.10,
            0.15,
            0.21,
            0.30,
            0.39,
            0.47,
            0.54,
            0.61,
            0.69,
            0.76,
            0.82,
            0.87,
            0.91,
            0.95,
            0.98,
        ],
        dtype=np.float64,
    )
    clean_closed = sample_curve_at_parameters(flower_curve, clustered_t, closed=True, noise_std=0.0, seed=26)
    clean_open = sample_curve(open_wave_curve, sample_count=18, closed=False, noise_std=0.0, seed=26)
    noisy_open = sample_curve(open_wave_curve, sample_count=24, closed=False, noise_std=0.06, seed=26)

    metric_rows: list[dict[str, str]] = []
    metric_rows.extend(plot_parameterization_comparison(clean_closed, output_dir / "parameterization_comparison.png"))
    metric_rows.extend(plot_method_comparison(clean_open, output_dir / "method_comparison.png"))
    metric_rows.extend(plot_noise_comparison(noisy_open, output_dir / "noise_robustness_comparison.png"))
    metric_rows.extend(plot_noise_robustness_curve(output_dir / "noise_sweep_comparison.png"))
    metric_rows.extend(build_metric_definition_rows(noisy_open))
    save_metrics_csv(metric_rows, output_dir / "metrics_summary.csv")

    print("Generated demo figures:")
    print(f"- {output_dir / 'parameterization_comparison.png'}")
    print(f"- {output_dir / 'method_comparison.png'}")
    print(f"- {output_dir / 'noise_robustness_comparison.png'}")
    print(f"- {output_dir / 'noise_sweep_comparison.png'}")
    print(f"- {output_dir / 'metrics_summary.csv'}")
    print()
    print("Metric summary")
    for row in metric_rows:
        print(
            f"{row['experiment']:>18} | {row['method']:<13} | {row['parameterization']:<11} "
            f"| sample_rmse={row['sample_rmse']} | chamfer={row['chamfer']}"
        )


def run_custom(args: argparse.Namespace) -> None:
    if args.input is None:
        raise ValueError("--input is required in custom mode")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    points = load_points(args.input)
    output_path = output_dir / "custom_curve_fit.png"
    plot_custom_fit(
        points=points,
        closed=args.closed,
        method=args.method,
        parameterization=args.parameterization,
        smooth_scale=args.smooth_scale,
        output_path=output_path,
    )
    print(f"Saved custom fitting result to: {output_path}")


def main() -> None:
    args = parse_args()
    if args.list_methods:
        print("Supported methods:")
        print("- interpolation: cubic spline interpolation")
        print("- pchip: piecewise cubic Hermite interpolation")
        print("- rbf: radial basis function interpolation")
        print("- smoothing: parametric smoothing spline")
        return
    if args.mode == "demo":
        run_demo(args.output_dir)
    else:
        run_custom(args)


if __name__ == "__main__":
    main()
