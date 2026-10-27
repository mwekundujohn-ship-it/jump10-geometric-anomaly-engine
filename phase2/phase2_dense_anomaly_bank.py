import numpy as np
from scipy.signal import hilbert

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_OK = True
except Exception:
    torch = None
    nn = None
    F = None
    TORCH_OK = False


FEATURE_NAMES = [
    "s_y", "int_t_dy", "int_y_dt", "levy_area", "quad_var", "cubic_var", "int_y2_dt", "int_t2_dy", "lead_lag_cov",
    "rqa_rr", "rqa_det", "rqa_lam", "rqa_diag_entropy",
    "ord_p0", "ord_p1", "ord_p2", "ord_p3", "ord_p4", "ord_p5", "perm_entropy", "forbidden_divergence",
    "arc_length", "elastic_energy", "mean_signed_curvature", "max_abs_curvature", "inflection_rate", "total_turning",
    "scat_s1_mean", "scat_s1_std", "scat_s1_2nd", "scat_s2_mean", "scat_s2_std", "scat_s2_2nd",
    "scat_s4_mean", "scat_s4_std", "scat_s4_2nd", "scat_s8_mean", "scat_s8_std", "scat_s8_2nd",
    "h0_max_gap", "h0_second_gap", "h0_persistence_entropy", "h1_loop_closure",
    "hurst_q1", "hurst_q2", "multifractal_width", "katz_dim", "higuchi_proxy",
    "hilbert_freq_mean", "hilbert_freq_std", "hilbert_amp_kurt",
    "hvg_edge_density", "hvg_degree_mean", "hvg_degree_std", "hvg_degree_max",
    "cumulant4", "trispectrum_mag", "trispectrum_phase",
    "emd_extrema_rate", "emd_envelope_asymmetry", "emd_zero_cross_rate",
    "dist_skew", "dist_kurtosis", "range_ratio",
]

N_FEATURES = len(FEATURE_NAMES)
assert N_FEATURES == 64, N_FEATURES


def _safe_div(a, b, eps=1e-9):
    return a / (b + eps)


def _entropy_from_counts(counts, axis=-1):
    total = np.sum(counts, axis=axis, keepdims=True)
    p = counts / np.maximum(total, 1e-9)
    return -np.sum(p * np.log(p + 1e-9), axis=axis, keepdims=True)


def _hvg_features(y):
    n, length = y.shape
    degree = np.zeros((n, length), dtype=np.float64)
    total_edges = np.zeros(n, dtype=np.float64)
    possible = 0
    max_lag = min(8, length - 1)
    for d in range(1, max_lag + 1):
        width = length - d + 1
        if d == 1:
            edge = np.ones((n, width), dtype=bool)
        else:
            mids = [y[:, i:i + width] for i in range(1, d)]
            middle_max = np.maximum.reduce(mids)
            edge = middle_max < np.minimum(y[:, :width], y[:, d - 1:])
        total_edges += edge.sum(axis=1)
        degree[:, :width] += edge
        degree[:, d - 1:] += edge
        possible += width
    possible = max(possible, 1)
    return np.column_stack([
        total_edges / possible,
        degree.mean(axis=1),
        degree.std(axis=1),
        degree.max(axis=1),
    ])


def _rqa_features(y):
    n, length = y.shape
    v0 = y[:, :-4]
    v1 = y[:, 2:-2]
    v2 = y[:, 4:]
    if v0.shape[1] < 4:
        rr = np.zeros((n, 1))
        det = np.zeros((n, 1))
        lam = np.zeros((n, 1))
        diag_entropy = np.zeros((n, 1))
        h1 = np.zeros((n, 1))
        return np.hstack([rr, det, lam, diag_entropy]), h1
    idx = np.linspace(0, v0.shape[1] - 1, min(16, v0.shape[1])).astype(int)
    v = np.stack([v0[:, idx], v1[:, idx], v2[:, idx]], axis=-1)
    dist_sq = np.sum((v[:, :, None, :] - v[:, None, :, :]) ** 2, axis=-1)
    eps = np.percentile(dist_sq, 25, axis=(1, 2), keepdims=True) + 1e-7
    rec = (dist_sq <= eps).astype(np.float64)
    eye = 1.0 - np.eye(rec.shape[1], dtype=np.float64)[None, :, :]
    rec_off = rec * eye
    total = rec_off.sum(axis=(1, 2), keepdims=True)
    rr = rec_off.mean(axis=(1, 2), keepdims=True)
    det_mat = rec_off[:, :-1, :-1] * rec_off[:, 1:, 1:]
    det = det_mat.sum(axis=(1, 2), keepdims=True)
    det = _safe_div(det, total)
    lam_mat = rec_off[:, :-1, :] * rec_off[:, 1:, :]
    lam = lam_mat.sum(axis=(1, 2), keepdims=True)
    lam = _safe_div(lam, total)
    diag_counts = []
    for lag in range(1, rec_off.shape[1]):
        c = np.sum(rec_off[:, :-lag, :-lag] * rec_off[:, lag:, lag:], axis=(1, 2))
        diag_counts.append(c[:, None])
    if diag_counts:
        diag_counts_arr = np.hstack(diag_counts)
    else:
        diag_counts_arr = np.zeros((n, 1))
    diag_entropy = _entropy_from_counts(diag_counts_arr, axis=1)
    h1 = np.min(dist_sq + np.eye(dist_sq.shape[1])[None, :, :] * 1e12, axis=(1, 2)).reshape(n, 1)
    return np.hstack([rr.reshape(n, 1), det.reshape(n, 1), lam.reshape(n, 1), diag_entropy.reshape(n, 1)]), h1


def _ordinal_features(y):
    w0 = y[:, :-2]
    w1 = y[:, 1:-1]
    w2 = y[:, 2:]
    c01 = (w0 < w1).astype(np.int32)
    c12 = (w1 < w2).astype(np.int32)
    c02 = (w0 < w2).astype(np.int32)
    code = c01 * 4 + c12 * 2 + c02
    valid = [0, 2, 3, 4, 5, 7]
    probs = np.stack([(code == c).mean(axis=1) for c in valid], axis=1)
    perm_entropy = -np.sum(probs * np.log(probs + 1e-9), axis=1, keepdims=True) / np.log(6.0)
    forbidden = np.sum((probs - (1.0 / 6.0)) ** 2, axis=1, keepdims=True)
    return np.hstack([probs, perm_entropy, forbidden])


def _scattering_features(y):
    parts = []
    for scale in (1, 2, 4, 8):
        if y.shape[1] <= scale:
            parts.extend([np.zeros((y.shape[0], 1))] * 3)
            continue
        d1 = np.abs(y[:, scale:] - y[:, :-scale])
        parts.append(d1.mean(axis=1, keepdims=True))
        parts.append(d1.std(axis=1, keepdims=True))
        if d1.shape[1] > 2:
            d2 = np.abs(d1[:, 2:] - d1[:, :-2])
            parts.append(d2.mean(axis=1, keepdims=True))
        else:
            parts.append(np.zeros((y.shape[0], 1)))
    return np.hstack(parts)


def compute_quartile_brute_force_features(y_zone):
    y = np.asarray(y_zone, dtype=np.float64)
    n, length = y.shape
    if length < 9:
        raise ValueError(f"zone too short: {length}")
    dy = np.diff(y, axis=1)
    ddy = np.diff(dy, axis=1)
    t = np.linspace(0.0, 1.0, length, dtype=np.float64)[None, :]
    dt = 1.0 / (length - 1)
    y0 = y - y[:, 0:1]

    s_y = y0[:, -1:]
    int_t_dy = np.sum(t[:, :-1] * dy, axis=1, keepdims=True)
    int_y_dt = np.sum(y0[:, :-1] * dt, axis=1, keepdims=True)
    levy_area = 0.5 * (int_t_dy - int_y_dt)
    quad_var = np.sum(dy ** 2, axis=1, keepdims=True)
    cubic_var = np.sum(dy ** 3, axis=1, keepdims=True)
    int_y2_dt = np.sum((y0[:, :-1] ** 2) * dt, axis=1, keepdims=True)
    int_t2_dy = np.sum((t[:, :-1] ** 2) * dy, axis=1, keepdims=True)
    lead_lag_cov = np.sum(dy[:, 1:] * dy[:, :-1], axis=1, keepdims=True)
    rough = np.hstack([s_y, int_t_dy, int_y_dt, levy_area, quad_var, cubic_var, int_y2_dt, int_t2_dy, lead_lag_cov])

    rqa, h1_loop = _rqa_features(y)
    ordinal = _ordinal_features(y)

    ds = np.sqrt(dt ** 2 + dy ** 2)
    arc_length = np.sum(ds, axis=1, keepdims=True)
    kappa = (dt * ddy) / ((dt ** 2 + dy[:, :-1] ** 2) ** 1.5 + 1e-9)
    elastic_energy = np.mean(kappa ** 2, axis=1, keepdims=True)
    mean_signed_curvature = np.mean(kappa, axis=1, keepdims=True)
    max_abs_curvature = np.max(np.abs(kappa), axis=1, keepdims=True)
    inflection_rate = np.mean((kappa[:, 1:] * kappa[:, :-1]) < 0, axis=1, keepdims=True)
    angles = np.arctan2(dy, dt)
    total_turning = np.sum(np.abs(np.diff(angles, axis=1)), axis=1, keepdims=True)
    frenet = np.hstack([arc_length, elastic_energy, mean_signed_curvature, max_abs_curvature, inflection_rate, total_turning])

    scattering = _scattering_features(y)

    abs_dy = np.sort(np.abs(dy), axis=1)
    h0_max = abs_dy[:, -1:]
    h0_second = abs_dy[:, -2:-1]
    h0_entropy = _entropy_from_counts(abs_dy, axis=1)
    hvg = _hvg_features(y)
    topology = np.hstack([h0_max.reshape(n, 1), h0_second.reshape(n, 1), h0_entropy.reshape(n, 1), h1_loop.reshape(n, 1)])

    tau1 = np.mean(np.abs(y[:, 1:] - y[:, :-1]), axis=1, keepdims=True) + 1e-7
    tau4 = np.mean(np.abs(y[:, 4:] - y[:, :-4]), axis=1, keepdims=True) + 1e-7
    hurst_q1 = np.log2(tau4 / tau1) / 2.0
    tau1_q2 = np.sqrt(np.mean((y[:, 1:] - y[:, :-1]) ** 2, axis=1, keepdims=True)) + 1e-7
    tau4_q2 = np.sqrt(np.mean((y[:, 4:] - y[:, :-4]) ** 2, axis=1, keepdims=True)) + 1e-7
    hurst_q2 = np.log2(tau4_q2 / tau1_q2) / 2.0
    multifractal_width = hurst_q1 - hurst_q2
    max_planar_dist = np.max(np.abs(y - y[:, 0:1]), axis=1, keepdims=True) + 1e-7
    katz_dim = np.log10(length) / (np.log10(max_planar_dist / (arc_length + 1e-7)) + np.log10(length) + 1e-7)
    higuchi_proxy = np.log(arc_length + 1e-7) / (np.log(max_planar_dist + 1e-7) + 1e-7)
    multifractal = np.hstack([hurst_q1, hurst_q2, multifractal_width, katz_dim, higuchi_proxy])

    analytic = hilbert(y0, axis=1)
    inst_amp = np.abs(analytic)
    inst_phase = np.unwrap(np.angle(analytic), axis=1)
    inst_freq = np.diff(inst_phase, axis=1)
    hfreq_mean = np.mean(inst_freq, axis=1, keepdims=True)
    hfreq_std = np.std(inst_freq, axis=1, keepdims=True)
    amp_kurt = np.mean(((inst_amp - inst_amp.mean(axis=1, keepdims=True)) / (inst_amp.std(axis=1, keepdims=True) + 1e-7)) ** 4, axis=1, keepdims=True)
    hilbert_feats = np.hstack([hfreq_mean, hfreq_std, amp_kurt])

    spectrum = np.fft.rfft(y, axis=1) / length
    tris = []
    for k1, k2 in ((1, 2), (1, 3), (2, 3)):
        if k1 + k2 < spectrum.shape[1]:
            val = spectrum[:, k1] * spectrum[:, k2] * np.conj(spectrum[:, k1 + k2])
            tris.append(np.abs(val)[:, None])
            tris.append(np.angle(val)[:, None])
    if tris:
        tris_arr = np.hstack(tris)
        tris_mag = np.mean(tris_arr[:, 0::2], axis=1, keepdims=True)
        tris_phase = np.mean(tris_arr[:, 1::2], axis=1, keepdims=True)
    else:
        tris_mag = np.zeros((n, 1))
        tris_phase = np.zeros((n, 1))
    centered = y - y.mean(axis=1, keepdims=True)
    std = y.std(axis=1, keepdims=True) + 1e-7
    z = centered / std
    cumulant4 = np.mean(z ** 4, axis=1, keepdims=True) - 3.0
    trispectrum = np.hstack([cumulant4.reshape(n, 1), tris_mag.reshape(n, 1), tris_phase.reshape(n, 1)])

    diff_sign = np.diff(np.signbit(y).astype(np.int8), axis=1)
    extremas = np.abs(np.diff(np.signbit(dy).astype(np.int8), axis=1)).sum(axis=1, keepdims=True)
    emd_extrema_rate = extremas / max(length - 2, 1)
    med = np.median(y, axis=1, keepdims=True)
    emd_env_asym = (y.max(axis=1, keepdims=True) - med) / (med - y.min(axis=1, keepdims=True) + 1e-7)
    emd_zero_cross = np.mean(diff_sign != 0, axis=1, keepdims=True)
    emd = np.hstack([emd_extrema_rate.reshape(n, 1), emd_env_asym.reshape(n, 1), emd_zero_cross.reshape(n, 1)])

    dist_skew = np.mean(z ** 3, axis=1, keepdims=True)
    dist_kurt = np.mean(z ** 4, axis=1, keepdims=True)
    range_ratio = (y.max(axis=1, keepdims=True) - y.min(axis=1, keepdims=True)) / std
    higher = np.hstack([dist_skew.reshape(n, 1), dist_kurt.reshape(n, 1), range_ratio.reshape(n, 1)])

    raw = np.hstack([
        rough, rqa, ordinal, frenet, scattering, topology, multifractal,
        hilbert_feats, hvg, trispectrum, emd, higher,
    ]).astype(np.float32)
    return np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)


if TORCH_OK:
    class LearnableGeometricAnomalyBank(nn.Module):
        def __init__(self, seq_len=38, num_wavelets=16, rqa_kernels=8, out_dim=128):
            super().__init__()
            self.num_wavelets = num_wavelets
            self.wavelet_freqs = nn.Parameter(torch.linspace(1.0, 18.0, num_wavelets))
            self.wavelet_scales = nn.Parameter(torch.ones(num_wavelets) * 0.15)
            self.wavelet_phases = nn.Parameter(torch.zeros(num_wavelets))
            self.delay_conv = nn.Conv1d(1, rqa_kernels, kernel_size=5, stride=1, dilation=2, padding=4)
            self.rqa_log_eps = nn.Parameter(torch.linspace(-2.0, 1.0, rqa_kernels))
            self.rqa_temp = nn.Parameter(torch.ones(rqa_kernels) * 5.0)
            self.lead_lag_bilinear = nn.Bilinear(8, 8, 32)
            self.path_proj = nn.Linear(1, 8)
            total = num_wavelets * 2 + rqa_kernels * 2 + 32 + 4
            self.fusion_net = nn.Sequential(
                nn.Linear(total, 256),
                nn.GELU(),
                nn.LayerNorm(256),
                nn.Linear(256, out_dim),
                nn.GELU(),
                nn.LayerNorm(out_dim),
            )

        def forward(self, y_ticks):
            b, length = y_ticks.shape
            t = torch.linspace(-0.5, 0.5, length, device=y_ticks.device)[None, None, :]
            freqs = self.wavelet_freqs[None, :, None]
            scales = torch.clamp(self.wavelet_scales[None, :, None].abs(), min=0.03)
            phases = self.wavelet_phases[None, :, None]
            envelope = torch.exp(-0.5 * (t / scales) ** 2)
            kernel_real = envelope * torch.cos(2.0 * np.pi * freqs * t + phases)
            kernel_imag = envelope * torch.sin(2.0 * np.pi * freqs * t + phases)
            y_exp = y_ticks[:, None, :]
            w_real = torch.mean(y_exp * kernel_real, dim=-1)
            w_imag = torch.mean(y_exp * kernel_imag, dim=-1)
            w_modulus = torch.sqrt(w_real ** 2 + w_imag ** 2 + 1e-8)
            w_phase = torch.atan2(w_imag, w_real + 1e-8)

            h_delay = self.delay_conv(y_exp)
            idx = torch.linspace(0, length - 1, min(12, length), device=y_ticks.device).long()
            h_sub = h_delay[:, :, idx]
            dist = (h_sub[:, :, :, None] - h_sub[:, :, None, :]) ** 2
            eps = torch.exp(self.rqa_log_eps)[None, :, None, None]
            temp = torch.clamp(self.rqa_temp.abs(), min=1.0)[None, :, None, None]
            soft_rec = torch.sigmoid(temp * (eps - dist))
            rqa_density = torch.mean(soft_rec, dim=(-1, -2))
            rqa_diag = torch.mean(soft_rec[:, :, :-1, :-1] * soft_rec[:, :, 1:, 1:], dim=(-1, -2))

            dy = y_ticks[:, 1:] - y_ticks[:, :-1]
            h_path = self.path_proj(dy[:, :, None])
            cum_path = torch.cumsum(h_path, dim=1) / length
            lead_vec = torch.mean(h_path[:, 1:, :], dim=1)
            lag_vec = torch.mean(cum_path[:, :-1, :], dim=1)
            rough_sig = self.lead_lag_bilinear(lead_vec, lag_vec)

            ddy = dy[:, 1:] - dy[:, :-1]
            dt = 1.0 / length
            kappa = (dt * ddy) / ((dt ** 2 + dy[:, :-1] ** 2) ** 1.5 + 1e-8)
            curv_energy = torch.mean(kappa ** 2, dim=-1, keepdim=True)
            curv_skew = torch.mean(kappa ** 3, dim=-1, keepdim=True)
            levy = 0.5 * (
                torch.sum(t[:, 0, :-1] * dy, dim=-1, keepdim=True)
                - torch.sum(y_ticks[:, :-1] * dt, dim=-1, keepdim=True)
            )
            drift = y_ticks[:, -1:] - y_ticks[:, 0:1]
            combined = torch.cat([
                w_modulus, w_phase, rqa_density, rqa_diag,
                rough_sig, curv_energy, curv_skew, levy, drift,
            ], dim=-1)
            return self.fusion_net(combined)
