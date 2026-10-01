#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Свободные затухающие колебания в контуре: спираль на фазовой плоскости, логарифмический
декремент Θ, добротность Q = π/Θ, критическое сопротивление R_cr (лаб. 3.2.5, пп. 2.3-2.4, 2.7).

Входные файлы: *.bin осциллографа АКТАКОМ ADS-6xxx (формат SPBXDS: заголовок JSON + int16),
имя файла = сопротивление R в омах (410.bin, 600.bin, ...).
В файлах записан только CH1 (напряжение на ёмкости U_C) во времени; ток I ∝ dU_C/dt
получается численным дифференцированием, поэтому спираль (U_C, dU_C/dt) строится программно.

Что делается для каждого файла:
  1. разбор заголовка (частота дискретизации, масштаб В/отсчёт), поиск импульса генератора
     (участок с ограничением АЦП) - свободные колебания берутся ПОСЛЕ него;
  2. фит затухающей косинусоиды  U = A·exp(-β t)·cos(ω t + φ) + c  ->  β, ω, Θ = 2πβ/ω;
  3. независимо: Θ по последовательным экстремумам (= пересечения спирали с осью U_C):
     ln|x_j| линейно по номеру экстремума, наклон = -Θ/2  (это формула (1) методички, МНК);
  4. Q = π/Θ; теоретическая Q = sqrt(L/C)/R_Σ.
Общий результат: R_L (из зависимости β(R)), R_cr из графика 1/Θ² = f(1/R_Σ²) (п. 2.7.2).

Запуск:  python analyze_spiral.py [--L 0.1 --C 6e-9] [--RL 50] [--no-show]
"""
import argparse
import glob
import os
import re
import struct
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter, find_peaks

plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.labelsize": 13,
                     "legend.fontsize": 11})

L_NOM = 0.100        # Гн (по заданию)
C_NOM = 6e-9         # Ф  (по заданию)
# имя файла -> реальное сопротивление, Ом (если в названии при записи ошиблись)
FILE_R_FIX = {1100.0: 1000.0}
SKIPS_US = (5, 10, 20, 40)   # мкс: варианты начала фита после импульса -> оценка систематики
SKIP_MAIN_US = 10


# ------------------------------- ЧТЕНИЕ .bin --------------------------------
def read_bin(path):
    """Возвращает dict: v (В, за вычетом ничего), raw, sr (Гц), meta (текст заголовка)."""
    d = open(path, "rb").read()
    if d[:6] != b"SPBXDS":
        raise ValueError(f"{path}: неизвестный формат (нет сигнатуры SPBXDS)")
    n = struct.unpack("<I", d[6:10])[0]
    hdr = d[10:10 + n].decode("latin-1")
    nb = struct.unpack("<I", d[10 + n:14 + n])[0]
    raw = np.frombuffer(d[14 + n:14 + n + nb], dtype="<i2").astype(float)

    def get(key, default=None):
        m = re.search(r'"%s":"?([^",}\]]+)"?' % key, hdr)
        return m.group(1) if m else default

    m = re.search(r"([\d.]+)\s*([kMG]?)S/s", get("Sample_Rate", ""))
    if not m:
        raise ValueError(f"{path}: не найдена частота дискретизации в заголовке")
    sr = float(m.group(1)) * {"": 1, "k": 1e3, "M": 1e6, "G": 1e9}[m.group(2)]
    m = re.search(r"([\d.]+)\s*([mu]?)v", get("Voltage_Rate", "0.125mv"), re.I)
    lsb = float(m.group(1)) * {"": 1, "m": 1e-3, "u": 1e-6}[m.group(2).lower()]
    m = re.match(r"(\d+)", get("Probe_Magnification", "1X") or "1X")
    probe = float(m.group(1)) if m else 1.0
    return dict(raw=raw, v=raw * lsb * probe, sr=sr, meta=hdr)


def pulse_end(raw, sr):
    """Конец импульса генератора: последняя группа отсчётов на пределе АЦП."""
    hi = raw.max() - 40
    idx = np.where(raw >= hi)[0]
    if len(idx) == 0:
        raise ValueError("импульс генератора (ограничение АЦП) не найден")
    groups = np.split(idx, np.where(np.diff(idx) > 5)[0] + 1)
    g = max(groups, key=len)
    return int(g[-1]), len(g) / sr


# ------------------------------ ФИТ / ЭКСТРЕМУМЫ ----------------------------
def damped(t, A, b, w, ph, c):
    return A * np.exp(-b * t) * np.cos(w * t + ph) + c


def fit_decay(seg, sr):
    """Фит затухающей косинусоиды. Возвращает (p, sigma_p, rms)."""
    t = np.arange(len(seg)) / sr
    wl = 31 if sr > 5e6 else 11
    sm = savgol_filter(seg, wl, 3)
    base = np.median(seg[-500:])
    zc = np.where(np.diff(np.sign(sm - base)) != 0)[0]
    w0 = np.pi / (np.median(np.diff(zc)) / sr) if len(zc) > 2 else 2 * np.pi * 6000
    best = None
    with np.errstate(all="ignore"):
        for ph0 in np.linspace(-np.pi, np.pi, 9):
            for b0 in (2e3, 5e3, 1e4, 2e4):
                try:
                    p, cv = curve_fit(damped, t, seg, p0=[0.7, b0, w0, ph0, 0.0], maxfev=20000)
                    r = float(np.sum((damped(t, *p) - seg) ** 2))
                    if np.isfinite(r) and (best is None or r < best[0]):
                        best = (r, p, cv)
                except Exception:
                    pass
    if best is None:
        raise RuntimeError("фит не сошёлся")
    r, p, cv = best
    return p, np.sqrt(np.abs(np.diag(cv))), np.sqrt(r / len(seg))


def theta_from_fit(p):
    return 2 * np.pi * p[1] / p[2]


def extrema_theta(seg, sr, p, rms):
    """Экстремумы U_C (пересечения спирали с осью U_C) и Θ по наклону ln|x_j|(j)."""
    x = seg - p[4]
    T = 2 * np.pi / p[2]
    wl = int(T * sr / 8) | 1
    sm = savgol_filter(x, max(wl, 5), 3)
    thr = max(5 * rms, 0.01)
    ip, _ = find_peaks(sm, prominence=thr)
    im, _ = find_peaks(-sm, prominence=thr)
    idx = np.sort(np.concatenate([ip, im]))
    ex = []
    h = max(wl // 2, 3)
    for i in idx:                       # уточнение вершины параболой по сырым данным
        lo, hi = max(i - h, 0), min(i + h + 1, len(x))
        tt = np.arange(lo, hi) - i
        c2 = np.polyfit(tt, x[lo:hi], 2)
        tv = -c2[1] / (2 * c2[0]) if c2[0] != 0 else 0.0
        ex.append((i + tv, np.polyval(c2, tv)))
    ex = np.array(ex)
    if len(ex) < 3:
        return ex, np.nan, np.nan
    j = np.arange(len(ex))
    coef, cov = np.polyfit(j, np.log(np.abs(ex[:, 1])), 1, cov=True) if len(ex) > 3 else (
        np.polyfit(j, np.log(np.abs(ex[:, 1])), 1), np.array([[np.nan, 0], [0, np.nan]]))
    return ex, -2 * coef[0], 2 * np.sqrt(abs(cov[0, 0]))


def spiral_xy(seg, sr, p):
    """Спираль: x = U_C - c, y = (dU_C/dt)/ω  (∝ току, в вольтах - чтобы спираль была круглой)."""
    T = 2 * np.pi / p[2]
    wl = int(T * sr / 3.5) | 1                    # окно ~T/3.5: подавляет шум квантования АЦП
    d = savgol_filter(seg, max(wl, 7), 3, deriv=1, delta=1.0 / sr)
    seg = savgol_filter(seg, max(wl // 2 | 1, 5), 3)
    return seg - p[4], d / p[2]


# -------------------------------- АНАЛИЗ ФАЙЛА ------------------------------
def analyze_file(path):
    rec = read_bin(path)
    sr, v, raw = rec["sr"], rec["v"], rec["raw"]
    end, pw = pulse_end(raw, sr)
    fits = {}
    for s in SKIPS_US:
        i0 = end + int(s * 1e-6 * sr)
        p, sp, rms = fit_decay(v[i0:], sr)
        fits[s] = (p, sp, rms)
    p, sp, rms = fits[SKIP_MAIN_US]
    i0 = end + int(SKIP_MAIN_US * 1e-6 * sr)
    seg = v[i0:]
    th = theta_from_fit(p)
    th_all = np.array([theta_from_fit(fits[s][0]) for s in SKIPS_US])
    # статистика фита: β и ω коррелированы, берём консервативно сумму относительных погрешностей
    s_fit = th * np.hypot(sp[1] / p[1], sp[2] / p[2])
    s_sys = float(np.std(th_all))
    ex, th_ex, s_ex = extrema_theta(seg, sr, p, rms)
    s_meth = abs(th - th_ex) / 2 if np.isfinite(th_ex) else 0.0   # расхождение двух способов определения Θ
    x, y = spiral_xy(seg, sr, p)
    return dict(rec=rec, sr=sr, end=end, i0=i0, seg=seg, p=p, sp=sp, rms=rms,
                theta=th, s_theta=float(np.sqrt(s_fit ** 2 + s_sys ** 2 + s_meth ** 2)), s_meth=s_meth, s_fit=s_fit, s_sys=s_sys,
                theta_ex=th_ex, s_theta_ex=s_ex, ex=ex, x=x, y=y,
                f_d=p[2] / (2 * np.pi), beta=p[1])


# ----------------------------------- MAIN -----------------------------------
def main():
    
    # --- НАСТРОЙКА ПУТЕЙ ---
    # Получаем абсолютный путь к папке, где лежит этот скрипт (scripts)
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    # Поднимаемся на уровень выше в lab_3.2.4+5
    BASE_DIR = os.path.dirname(SCRIPT_DIR)
    
    # Формируем пути к данным и графикам
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    FIGURES_DIR = os.path.join(BASE_DIR, 'figures')
    
    # Создаем папку figures, если ее еще нет
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    ap = argparse.ArgumentParser()
    # Теперь по умолчанию аргумент --dir берет данные из папки data
    ap.add_argument("--dir", default=DATA_DIR)
    ap.add_argument("--out", default="spiral", help="префикс выходных файлов")
    ap.add_argument("--L", type=float, default=L_NOM, help="индуктивность, Гн")
    ap.add_argument("--C", type=float, default=C_NOM, help="ёмкость (по магазину), Ф")
    ap.add_argument("--RL", type=float, default=None,
                    help="R_L катушки, Ом (по LCR-метру). Без ключа - оценивается по β(R)")
    ap.add_argument("--R-err", type=float, default=0.01, dest="R_err",
                    help="относит. погрешность установки R на магазине (по умолч. 0.01 - ОЦЕНКА, уточнить по паспорту МСР-60)")
    ap.add_argument("--RL-err", type=float, default=5.0, dest="RL_err",
                    help="погрешность R_L, Ом, если R_L задан ключом --RL (по умолч. 5)")
    ap.add_argument("--set-R", nargs="*", default=[], metavar="ИМЯ=ОМ",
                    help="исправить сопротивление для файла, напр. --set-R 1100=1000")
    ap.add_argument("--auto-exclude", action="store_true",
                    help="автоматически исключать точки, выпадающие из линейной зависимости β(R) "
                         "(по умолчанию все точки используются)")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()
    override = dict(FILE_R_FIX)
    override.update({float(k): float(v) for k, v in (kv.split("=") for kv in args.set_R)})
    if args.no_show:
        matplotlib.use("Agg")
    L, C = args.L, args.C

    # Формируем базовое имя для сохранения в папке figures
    out_prefix = os.path.join(FIGURES_DIR, args.out)

    files = sorted(glob.glob(os.path.join(args.dir, "*.bin")),
                   key=lambda f: float(re.sub(r"\D", "", os.path.basename(f)) or 0))
    res = {}
    for f in files:
        m = re.match(r"(\d+)", os.path.basename(f))
        if not m:
            continue
        R = float(m.group(1))
        try:
            res[override.get(R, R)] = analyze_file(f)
        except Exception as e:
            print(f"[!] {os.path.basename(f)}: {e}", file=sys.stderr)
    if len(res) < 3:
        sys.exit(f"Найдено меньше 3 подходящих .bin файлов в директории {args.dir} (имя = сопротивление в омах, напр. 410.bin)")
    Rs = np.array(sorted(res))
    beta = np.array([res[R]["beta"] for R in Rs])
    w_d = np.array([2 * np.pi * res[R]["f_d"] for R in Rs])

    # ---- R_L и собственная частота ----
    # β = (R + R_L)/(2 L_eff): линейная зависимость; точки, выпадающие из неё, исключаются
    ok = np.ones(len(Rs), bool)
    for _ in range(4):
        k_b, b_b = np.polyfit(Rs[ok], beta[ok], 1)
        resid = beta - (k_b * Rs + b_b)
        sig = 1.4826 * np.median(np.abs(resid[ok] - np.median(resid[ok]))) + 1e-9
        new_ok = np.abs(resid) < max(6 * sig, 0.02 * beta.mean())
        if not args.auto_exclude:
            break
        if (new_ok == ok).all():
            break
        ok = new_ok
    R_L_fit, L_eff = b_b / k_b, 1 / (2 * k_b)
    (_, _), cv_b = np.polyfit(Rs[ok], beta[ok], 1, cov=True)
    s_RL_fit = abs(R_L_fit) * np.hypot(np.sqrt(cv_b[1, 1]) / abs(b_b), np.sqrt(cv_b[0, 0]) / abs(k_b))
    R_L = args.RL if args.RL is not None else R_L_fit
    s_RL = args.RL_err if args.RL is not None else s_RL_fit
    w0 = np.sqrt(w_d ** 2 + beta ** 2)            # собственная частота (не должна зависеть от R)
    w0m = float(np.mean(w0[ok]))
    C_eff = 1 / (L_eff * w0m ** 2)                # с учётом L_eff (согласовано с наклоном β(R))
    C_eff_nomL = 1 / (L * w0m ** 2)
    R_S = Rs + R_L
    s_RS = np.hypot(args.R_err * Rs, s_RL)         # погрешность R_Σ = R + R_L
    for i in np.where(~ok)[0]:
        R_pred = (beta[i] / k_b) - R_L_fit
        print(f"[!] файл {Rs[i]:.0f}.bin: β={beta[i]:.0f} 1/с не согласуется с остальными точками "
              f"(по ним это соответствует R ≈ {R_pred:.0f} Ом, а не {Rs[i]:.0f}). "
              f"Точка исключена из R_L и R_cr; проверьте номинал или запустите с --set-R {Rs[i]:.0f}={round(R_pred, -1):.0f}\n")

    # ---- Θ, Q ----
    th = np.array([res[R]["theta"] for R in Rs])
    s_th = np.array([res[R]["s_theta"] for R in Rs])
    th_ex = np.array([res[R]["theta_ex"] for R in Rs])
    Q = np.pi / th
    s_Q = np.pi * s_th / th ** 2
    Q_th6 = np.sqrt(L / C) / R_S
    Q_thE = w0m * L_eff / R_S

    # ---- R_cr: Y = 1/Θ², X = 1/R_Σ²;  Y = (L/C)/π² · X - 1/(4π²);  R_cr = 2π sqrt(ΔY/ΔX) ----
    X = 1 / R_S ** 2
    Y = 1 / th ** 2
    sY0 = 2 * s_th / th ** 3
    sX = 2 * s_RS / R_S ** 3
    k_y = np.polyfit(X[ok], Y[ok], 1)[0]
    for _ in range(4):                              # метод эффективной дисперсии (погрешности по X и Y)
        sY = np.sqrt(sY0 ** 2 + (k_y * sX) ** 2)
        (k_y, b_y), cov = np.polyfit(X[ok], Y[ok], 1, w=1 / sY[ok], cov="unscaled")
    chi2 = np.sum(((Y[ok] - (k_y * X[ok] + b_y)) / sY[ok]) ** 2) / max(ok.sum() - 2, 1)
    s_k = np.sqrt(cov[0, 0] * max(chi2, 1.0))
    Rcr = 2 * np.pi * np.sqrt(k_y)
    s_Rcr = np.pi * s_k / np.sqrt(k_y)
    Rcr_6 = 2 * np.sqrt(L / C)
    Rcr_E = 2 * w0m * L_eff

    # -------------------------------- ВЫВОД -----------------------------------
    print(f"L = {L*1e3:.1f} мГн, C = {C*1e9:.2f} нФ (заданы); файлов: {len(Rs)}\n")
    print("R_L: " + (f"задано {R_L:.1f} ± {s_RL:.1f} Ом" if args.RL is not None else
                     f"оценка по β(R): {R_L_fit:.1f} ± {s_RL_fit:.1f} Ом (при этом L_eff = {L_eff*1e3:.1f} мГн)"))
    print(f"погрешности: R магазина {100*args.R_err:.1f} % (оценка), Θ = фит + систематика по началу окна + "
          f"расхождение фит/экстремумы (1σ); на графиках ±2σ")
    print(f"собственная частота: f0 = {w0m/2/np.pi:.0f} Гц (разброс по R {np.std(w0[ok])/w0m*100:.2f} %)")
    print(f"L_eff = {L_eff*1e3:.1f} мГн (номинал {L*1e3:.0f}); C_eff = 1/(L_eff·ω0²) = {C_eff*1e9:.2f} нФ "
          f"(с L=100 мГн: {C_eff_nomL*1e9:.2f} нФ) -> паразитная ёмкость C0 ≈ {(C_eff-C)*1e9:.2f} нФ\n")
    hdr = (f"{'R, Ом':>7}{'R_Σ, Ом':>9}{'f_d, Гц':>9}{'β, 1/с':>9}{'Θ (фит)':>10}{'±1σ':>7}"
           f"{'Θ (экстр.)':>12}{'Q=π/Θ':>8}{'±1σ':>6}{'Q(6нФ)':>8}{'Q(C_eff)':>9}")
    print(hdr)
    print("-" * len(hdr))
    for i, R in enumerate(Rs):
        r = res[R]
        print(("*" if not ok[i] else " ") + f"{R:>6.0f}{R_S[i]:>9.1f}{r['f_d']:>9.1f}{r['beta']:>9.0f}{th[i]:>10.4f}{s_th[i]:>7.4f}"
              f"{th_ex[i]:>12.4f}{Q[i]:>8.2f}{s_Q[i]:>6.2f}{Q_th6[i]:>8.2f}{Q_thE[i]:>9.2f}")
    print("(* - точка исключена из R_L и R_cr)")
    print(f"\nR_cr по графику 1/Θ²=f(1/R_Σ²): {Rcr:.0f} ± {s_Rcr:.0f} Ом  (наклон {k_y:.4g}, "
          f"пересечение {b_y:.4f}; теория -1/(4π²) = {-1/(4*np.pi**2):.4f})")
    print(f"R_cr теория 2√(L/C):  C = {C*1e9:.1f} нФ -> {Rcr_6:.0f} Ом;  C_eff = {C_eff*1e9:.2f} нФ, L_eff -> 2·L_eff·ω0 = {Rcr_E:.0f} Ом")

    # -------------------------------- ГРАФИКИ ---------------------------------
    cm = plt.cm.viridis(np.linspace(0.05, 0.9, len(Rs)))
    fig, ax = plt.subplots(2, 2, figsize=(14, 11))
    a = ax[0, 0]
    for c_, R in zip(cm, Rs):
        r = res[R]
        a.plot(r["x"], r["y"], lw=0.9, color=c_, label=f"{R:.0f} Ом")
        if len(r["ex"]):
            xs = r["ex"][:, 1]
            a.plot(xs, np.zeros_like(xs), "o", ms=3.5, color=c_, mec="k", mew=0.4)
    a.axhline(0, color="gray", lw=0.6); a.axvline(0, color="gray", lw=0.6)
    a.set_aspect("equal", "datalim")
    a.set_xlabel("$U_C$, В"); a.set_ylabel("$\\dot U_C/\\omega$, В  (∝ ток)")
    a.set_title("Спираль на фазовой плоскости (точки на оси - экстремумы $U_C$)")
    a.legend(ncol=2, loc="upper right", fontsize=9); a.grid(True, ls="--", alpha=0.5)
    rms_all = np.mean([res[R]["rms"] for R in Rs])
    a.text(0.02, 0.02, f"шум по $U_C$: ±2σ ≈ ±{2*rms_all*1e3:.0f} мВ (мало в масштабе рисунка)",
           transform=a.transAxes, fontsize=9, alpha=0.8)

    a = ax[0, 1]
    for c_, R in zip(cm, Rs):
        ex = res[R]["ex"]
        if len(ex) >= 2:
            j = np.arange(len(ex))
            a.errorbar(j, np.abs(ex[:, 1]), yerr=2 * res[R]["rms"], fmt="o", color=c_, ms=5, ecolor="k",
                       elinewidth=0.8, capsize=2)
            a.set_yscale("log")
            a.semilogy(j, np.abs(ex[0, 1]) * np.exp(-res[R]["theta_ex"] / 2 * j), "-", color=c_, lw=1,
                       label=f"{R:.0f} Ом: Θ={res[R]['theta_ex']:.2f}")
    a.set_xlabel("номер экстремума j (половина периода)"); a.set_ylabel("$|x_j|$, В")
    a.set_title("Затухание экстремумов: наклон = −Θ/2 (усы ±2σ шума АЦП)")
    a.legend(fontsize=8, ncol=2); a.grid(True, which="both", ls="--", alpha=0.5)

    a = ax[1, 0]
    a.errorbar(X[ok] * 1e6, Y[ok], xerr=2 * sX[ok] * 1e6, yerr=2 * sY[ok], fmt="o", color="C0", mec="k",
               ecolor="k", elinewidth=1.3, capsize=4, label="эксперимент (±2σ по X и Y)")
    if (~ok).any():
        a.plot(X[~ok] * 1e6, Y[~ok], "rx", ms=9, mew=2, label="исключено (не согласуется по β)")
    xx = np.linspace(0, X.max() * 1.05, 50)
    a.plot(xx * 1e6, k_y * xx + b_y, "C3-", label=f"МНК: $R_{{cr}}$ = {Rcr:.0f} ± {s_Rcr:.0f} Ом")
    a.plot(xx * 1e6, (L / C / np.pi ** 2) * xx - 1 / (4 * np.pi ** 2), "k--", lw=1,
           label=f"теория (C = {C*1e9:.0f} нФ): $R_{{cr}}$ = {Rcr_6:.0f} Ом")
    a.set_xlabel("$1/R_\\Sigma^2$, $10^{-6}$ Ом$^{-2}$"); a.set_ylabel("$1/\\Theta^2$")
    a.set_title("Критическое сопротивление"); a.legend(fontsize=10); a.grid(True, ls="--", alpha=0.5)

    a = ax[1, 1]
    a.errorbar(Rs[ok], Q[ok], xerr=2 * s_RS[ok], yerr=2 * s_Q[ok], fmt="o", color="C0", mec="k", ecolor="k",
               elinewidth=1.3, capsize=4, label="π/Θ (фит, ±2σ)")
    if (~ok).any():
        a.plot(Rs[~ok], Q[~ok], "rx", ms=9, mew=2, label="исключено")
    a.plot(Rs, np.pi / th_ex, "s", color="C1", mfc="none", label="π/Θ (экстремумы)")
    rr = np.linspace(Rs.min() * 0.9, Rs.max() * 1.05, 100)
    a.plot(rr, np.sqrt(L / C) / (rr + R_L), "k--", label=f"$\\sqrt{{L/C}}/R_\\Sigma$, C = {C*1e9:.0f} нФ")
    a.plot(rr, w0m * L_eff / (rr + R_L), "C3-", label=f"$\\omega_0 L_{{eff}}/R_\\Sigma$, C = {C_eff*1e9:.1f} нФ")
    a.set_xlabel("R, Ом"); a.set_ylabel("Q"); a.set_title("Добротность")
    a.legend(fontsize=10); a.grid(True, ls="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_prefix + "_summary.png", dpi=150)
    fig.savefig(out_prefix + "_summary.pdf")

    # осциллограммы с фитом
    n = len(Rs)
    fig2, ax2 = plt.subplots((n + 1) // 2, 2, figsize=(14, 3.0 * ((n + 1) // 2) + 1), squeeze=False)
    for a, R in zip(ax2.flat, Rs):
        r = res[R]
        t = (np.arange(len(r["rec"]["v"])) - r["end"]) / r["sr"] * 1e3
        a.plot(t, r["rec"]["v"], lw=0.5, color="0.5", label="данные (CH1)")
        tt = np.arange(len(r["seg"])) / r["sr"]
        a.plot(t[r["i0"]:], damped(tt, *r["p"]), "C3-", lw=1.1, label="фит")
        a.fill_between(t[r["i0"]:], damped(tt, *r["p"]) - 2 * r["rms"], damped(tt, *r["p"]) + 2 * r["rms"],
                       color="C3", alpha=0.25, lw=0, label="±2σ (остаток фита)")
        if len(r["ex"]):
            a.plot((r["ex"][:, 0] + r["i0"] - r["end"]) / r["sr"] * 1e3, r["ex"][:, 1] + r["p"][4], "ko", ms=3)
        a.set_xlim(-0.05, min(t[-1], 5 / max(r["beta"], 1) * 1e3 * 1.6 + 0.2))
        a.set_title(f"R = {R:.0f} Ом:  Θ = {r['theta']:.3f} ± {r['s_theta']:.3f},  f = {r['f_d']:.0f} Гц")
        a.set_xlabel("t после импульса, мс"); a.set_ylabel("$U_C$, В"); a.grid(True, ls="--", alpha=0.5)
    ax2.flat[0].legend(fontsize=9)
    fig2.tight_layout()
    fig2.savefig(out_prefix + "_waveforms.png", dpi=130)
    fig2.savefig(out_prefix + "_waveforms.pdf")
    print(f"\nГрафики сохранены: {out_prefix}_summary.(png|pdf), {out_prefix}_waveforms.(png|pdf)")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()