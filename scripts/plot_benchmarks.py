#!/usr/bin/env python3
"""Generate clean SVG benchmark charts for Readme.md (zero external dependencies)."""

from pathlib import Path


def svg_header(width: int, height: int, title: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{title}">\n'
        f'  <rect width="{width}" height="{height}" rx="10" fill="#ffffff" stroke="#d0d7de" stroke-width="1"/>\n'
        f'  <style>\n'
        f'    .title {{ font: 700 16px -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; fill: #1f2328; }}\n'
        f'    .subtitle {{ font: 400 12px -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; fill: #656d76; }}\n'
        f'    .axis {{ font: 400 11px -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; fill: #656d76; }}\n'
        f'    .label {{ font: 600 11px -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; fill: #1f2328; }}\n'
        f'    .legend {{ font: 500 11.5px -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; fill: #1f2328; }}\n'
        f'  </style>\n'
    )


def generate_dgemm_chart(out_path: Path) -> None:
    width, height = 820, 420
    pad_l, pad_r, pad_t, pad_b = 70, 210, 68, 56
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    sizes = [256, 512, 1024, 2000, 2048]
    series = [
        ("naive_ijk", "#8b949e", [(256, 2.18), (512, 1.86), (1024, 1.76)]),
        ("fast_ikj", "#0969da", [(256, 13.5), (512, 13.5), (1024, 13.6), (2000, 13.3), (2048, 13.1)]),
        ("ikj_k4", "#8250df", [(256, 17.1), (512, 16.4), (1024, 17.0), (2000, 12.8), (2048, 12.5)]),
        ("blocked_k4_ikj (BS=64)", "#bf8700", [(256, 21.9), (512, 15.0), (1024, 14.7), (2000, 26.9), (2048, 13.0)]),
        ("neon_4x4 (BS=64)", "#1a7f37", [(256, 28.7), (512, 28.8), (1024, 29.1), (2000, 25.9), (2048, 25.1)]),
    ]

    max_y = 35.0
    x_pos = {s: pad_l + i * (plot_w / (len(sizes) - 1)) for i, s in enumerate(sizes)}

    def y_pos(val: float) -> float:
        return pad_t + plot_h - (val / max_y) * plot_h

    lines = [svg_header(width, height, "DGEMM Throughput Scaling")]
    lines.append(f'  <text x="{pad_l}" y="30" class="title">DGEMM Throughput Scaling (Custom C++ &amp; ARM64 NEON Kernels)</text>\n')
    lines.append(f'  <text x="{pad_l}" y="48" class="subtitle">Apple M2 Pro (single-threaded, -O3, median of 5 runs). Reference: accelerate_dgemm reaches 307–361 GFLOP/s.</text>\n')

    # Horizontal grid
    for tick in range(0, 36, 5):
        y = y_pos(tick)
        lines.append(f'  <line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l + plot_w}" y2="{y:.1f}" stroke="#eaeef2" stroke-width="1"/>\n')
        lines.append(f'  <text x="{pad_l - 10}" y="{y + 4:.1f}" text-anchor="end" class="axis">{tick}</text>\n')

    lines.append(f'  <text x="20" y="{pad_t + plot_h / 2:.1f}" transform="rotate(-90 20,{pad_t + plot_h / 2:.1f})" text-anchor="middle" class="axis">GFLOP/s (higher is better)</text>\n')

    # X axis labels
    for s in sizes:
        x = x_pos[s]
        lines.append(f'  <line x1="{x:.1f}" y1="{pad_t}" x2="{x:.1f}" y2="{pad_t + plot_h}" stroke="#f6f8fa" stroke-width="1"/>\n')
        lines.append(f'  <text x="{x:.1f}" y="{pad_t + plot_h + 20}" text-anchor="middle" class="axis">N={s}</text>\n')
    lines.append(f'  <text x="{pad_l + plot_w / 2:.1f}" y="{height - 12}" text-anchor="middle" class="axis">Matrix Dimension (N × N)</text>\n')

    # Axes lines
    lines.append(f'  <line x1="{pad_l}" y1="{pad_t + plot_h}" x2="{pad_l + plot_w}" y2="{pad_t + plot_h}" stroke="#8c959f" stroke-width="1.2"/>\n')
    lines.append(f'  <line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t + plot_h}" stroke="#8c959f" stroke-width="1.2"/>\n')

    # Series lines & points
    for idx, (name, color, pts) in enumerate(series):
        coords = [(x_pos[n], y_pos(v)) for n, v in pts]
        path_d = " ".join(("M" if i == 0 else "L") + f" {x:.1f} {y:.1f}" for i, (x, y) in enumerate(coords))
        lines.append(f'  <path d="{path_d}" fill="none" stroke="{color}" stroke-width="2.5" stroke-linejoin="round"/>\n')
        for x, y in coords:
            lines.append(f'  <circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}" stroke="#ffffff" stroke-width="1.5"/>\n')

        # Legend entry
        lx = pad_l + plot_w + 20
        ly = pad_t + 20 + idx * 28
        lines.append(f'  <line x1="{lx}" y1="{ly}" x2="{lx + 20}" y2="{ly}" stroke="{color}" stroke-width="2.5"/>\n')
        lines.append(f'  <circle cx="{lx + 10}" cy="{ly}" r="3.5" fill="{color}"/>\n')
        lines.append(f'  <text x="{lx + 28}" y="{ly + 4}" class="legend">{name}</text>\n')

    lines.append('</svg>\n')
    out_path.write_text("".join(lines), encoding="utf-8")


def generate_opencv_conv_chart(out_path: Path) -> None:
    width, height = 760, 340
    pad_l, pad_r, pad_t, pad_b = 185, 70, 68, 48
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    items = [
        ("conv2d_naive (4-loop)", 2.60, "162 ms", "#8b949e"),
        ("im2col + ikj_f32", 8.13, "51.7 ms", "#0969da"),
        ("OpenCV cv::filter2D", 39.5, "10.6 ms", "#bf8700"),
        ("im2col + NEON 4×4 f32", 40.7, "10.3 ms", "#1a7f37"),
    ]

    max_x = 45.0
    row_h = plot_h / len(items)
    bar_h = 28

    lines = [svg_header(width, height, "2D Convolution Throughput")]
    lines.append('  <text x="24" y="30" class="title">2D Convolution via im2col + GEMM vs OpenCV cv::filter2D</text>\n')
    lines.append('  <text x="24" y="48" class="subtitle">512 × 512 image (CV_32F), 5 × 5 kernel, 32 filters — Apple M2 Pro (-O3, median of 5 runs)</text>\n')

    for tick in range(0, 46, 10):
        x = pad_l + (tick / max_x) * plot_w
        lines.append(f'  <line x1="{x:.1f}" y1="{pad_t}" x2="{x:.1f}" y2="{pad_t + plot_h}" stroke="#eaeef2" stroke-width="1"/>\n')
        lines.append(f'  <text x="{x:.1f}" y="{pad_t + plot_h + 18}" text-anchor="middle" class="axis">{tick}</text>\n')

    lines.append(f'  <text x="{pad_l + plot_w / 2:.1f}" y="{height - 10}" text-anchor="middle" class="axis">Throughput (GFLOP/s, higher is better)</text>\n')
    lines.append(f'  <line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t + plot_h}" stroke="#8c959f" stroke-width="1.2"/>\n')

    for i, (label, gflops, ms, color) in enumerate(items):
        cy = pad_t + i * row_h + row_h / 2
        by = cy - bar_h / 2
        bw = (gflops / max_x) * plot_w
        lines.append(f'  <text x="{pad_l - 12}" y="{cy + 4:.1f}" text-anchor="end" class="legend">{label}</text>\n')
        lines.append(f'  <rect x="{pad_l}" y="{by:.1f}" width="{bw:.1f}" height="{bar_h}" rx="4" fill="{color}"/>\n')
        lines.append(f'  <text x="{pad_l + bw + 8:.1f}" y="{cy + 4:.1f}" class="label">{gflops:.1f} GFLOP/s ({ms})</text>\n')

    lines.append('</svg>\n')
    out_path.write_text("".join(lines), encoding="utf-8")


def generate_opencv_gemm_chart(out_path: Path) -> None:
    width, height = 760, 320
    pad_l, pad_r, pad_t, pad_b = 185, 70, 68, 48
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    items = [
        ("neon_4x4 (CV_64F)", 27.2, "79.3 ms", "#1a7f37"),
        ("OpenCV cv::gemm", 358.7, "9.42 ms", "#bf8700"),
        ("Apple Accelerate", 327.5, "6.58 ms", "#0969da"),
    ]

    max_x = 400.0
    row_h = plot_h / len(items)
    bar_h = 30

    lines = [svg_header(width, height, "Dense cv::Mat GEMM Comparison")]
    lines.append('  <text x="24" y="30" class="title">Dense cv::Mat GEMM: Custom NEON vs OpenCV cv::gemm vs Accelerate</text>\n')
    lines.append('  <text x="24" y="48" class="subtitle">1024 × 1024 double-precision (CV_64F) — Apple M2 Pro (-O3, median of 5 runs)</text>\n')

    for tick in range(0, 401, 100):
        x = pad_l + (tick / max_x) * plot_w
        lines.append(f'  <line x1="{x:.1f}" y1="{pad_t}" x2="{x:.1f}" y2="{pad_t + plot_h}" stroke="#eaeef2" stroke-width="1"/>\n')
        lines.append(f'  <text x="{x:.1f}" y="{pad_t + plot_h + 18}" text-anchor="middle" class="axis">{tick}</text>\n')

    lines.append(f'  <text x="{pad_l + plot_w / 2:.1f}" y="{height - 10}" text-anchor="middle" class="axis">Throughput (GFLOP/s, higher is better)</text>\n')
    lines.append(f'  <line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t + plot_h}" stroke="#8c959f" stroke-width="1.2"/>\n')

    for i, (label, gflops, ms, color) in enumerate(items):
        cy = pad_t + i * row_h + row_h / 2
        by = cy - bar_h / 2
        bw = (gflops / max_x) * plot_w
        lines.append(f'  <text x="{pad_l - 12}" y="{cy + 4:.1f}" text-anchor="end" class="legend">{label}</text>\n')
        lines.append(f'  <rect x="{pad_l}" y="{by:.1f}" width="{bw:.1f}" height="{bar_h}" rx="4" fill="{color}"/>\n')
        lines.append(f'  <text x="{pad_l + bw + 8:.1f}" y="{cy + 4:.1f}" class="label">{gflops:.1f} GFLOP/s ({ms})</text>\n')

    lines.append('</svg>\n')
    out_path.write_text("".join(lines), encoding="utf-8")


def generate_speedup_chart(out_path: Path) -> None:
    width, height = 760, 360
    pad_l, pad_r, pad_t, pad_b = 185, 95, 68, 48
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    items = [
        ("naive_ijk", 1.0, "1218 ms", "#8b949e"),
        ("tiled_bs64", 5.3, "228 ms", "#6e7781"),
        ("fast_ikj", 7.7, "158 ms", "#0969da"),
        ("ikj_k4", 9.6, "127 ms", "#8250df"),
        ("neon_4x4 (bs64)", 16.5, "73.9 ms", "#2da44e"),
        ("neon_4x4 (bs16)", 18.8, "64.9 ms", "#1a7f37"),
    ]

    max_x = 22.0
    row_h = plot_h / len(items)
    bar_h = 26

    lines = [svg_header(width, height, "Speedup vs Naive i-j-k at N=1024")]
    lines.append('  <text x="24" y="30" class="title">Speedup vs Naive i-j-k Baseline (N = 1024, Double Precision)</text>\n')
    lines.append('  <text x="24" y="48" class="subtitle">Apple M2 Pro (-O3, median of 5 runs). Reference: accelerate_dgemm reaches 191.8× (6.35 ms).</text>\n')

    for tick in range(0, 21, 5):
        x = pad_l + (tick / max_x) * plot_w
        lines.append(f'  <line x1="{x:.1f}" y1="{pad_t}" x2="{x:.1f}" y2="{pad_t + plot_h}" stroke="#eaeef2" stroke-width="1"/>\n')
        lines.append(f'  <text x="{x:.1f}" y="{pad_t + plot_h + 18}" text-anchor="middle" class="axis">{tick}×</text>\n')

    lines.append(f'  <text x="{pad_l + plot_w / 2:.1f}" y="{height - 10}" text-anchor="middle" class="axis">Speedup (higher is better)</text>\n')
    lines.append(f'  <line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t + plot_h}" stroke="#8c959f" stroke-width="1.2"/>\n')

    for i, (label, sp, ms, color) in enumerate(items):
        cy = pad_t + i * row_h + row_h / 2
        by = cy - bar_h / 2
        bw = (sp / max_x) * plot_w
        lines.append(f'  <text x="{pad_l - 12}" y="{cy + 4:.1f}" text-anchor="end" class="legend">{label}</text>\n')
        lines.append(f'  <rect x="{pad_l}" y="{by:.1f}" width="{bw:.1f}" height="{bar_h}" rx="4" fill="{color}"/>\n')
        lines.append(f'  <text x="{pad_l + bw + 8:.1f}" y="{cy + 4:.1f}" class="label">{sp:.1f}× ({ms})</text>\n')

    lines.append('</svg>\n')
    out_path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "docs" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    generate_dgemm_chart(out_dir / "gflops_scaling.svg")
    generate_speedup_chart(out_dir / "speedup_comparison.svg")
    generate_opencv_conv_chart(out_dir / "opencv_conv2d.svg")
    generate_opencv_gemm_chart(out_dir / "opencv_cv_gemm.svg")
    print(f"Generated SVG charts in {out_dir}")


if __name__ == "__main__":
    main()
