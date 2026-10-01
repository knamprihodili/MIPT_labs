#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Лаб. 3.2.5 (4.6), п. 2.7.7-2.7.13: резонансные кривые U/U0 = f(ν/ν0), добротность по АЧХ и ФЧХ,
погрешности, сводная таблица Q всеми способами.

Используются результаты уже готовых скриптов:
  analyze_rlc.py     - АЧХ (r1.csv, r2.csv) и ФЧХ (fchh410.csv, fchh2040.csv) при развёртке частоты;
  analyze_spiral.py  - свободные колебания (410.bin, 2040.bin): Θ по фиту и по экстремумам (спираль).
Сопротивления: R1 = 410 Ом, R2 = 2040 Ом (0.05 и 0.25 от R_cr).

Физическая модель контура (та же, что в установке, рис. 1-2 методички):
  генератор -> C1 -> [ C  ||  (R + R_L + L) ] ,   U_C измеряется на C.
  H(f) = Y/(1+Y),  Y = jωC1·Z_t,  Z_t = (R_Σ + jωL) / (1 - ω²LC + jωR_Σ C)
Одна модель с общими L, C, C1, R_L, U_g аппроксимирует ВСЕ ЧЕТЫРЕ кривые (две АЧХ и две ФЧХ).
Она нужна как гладкая интерполяция через участки вблизи частот k·500 Гц, где данные с осциллографа
отброшены (артефакт дискретизации), и через диапазон, в который измерение не дошло. Все операции
методички (положение максимума, ширина на уровне 1/√2, уровни -π/4, -π/2, -3π/4) выполняются
на этой кривой, а затем сверяются с прямым измерением по экспериментальным точкам.

Запуск:
  python analyze_q.py --dir ПАПКА_С_ДАННЫМИ [--nboot 120] [--no-show]
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.signal import savgol_filter
from scipy.interpolate import PchipInterpolator

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyze_rlc as rlc          # noqa: E402
import analyze_spiral_finita as spi       # noqa: E402

plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.labelsize": 13, "legend.fontsize": 11})

DATA_DIR = r"C:\Utilities\Projects\MIPT_labs\lab_3.2.4+5\data"
FIGURES_DIR = r"C:\Utilities\Projects\MIPT_labs\lab_3.2.4+5\figures"

# ------------------------------ ИСХОДНЫЕ ДАННЫЕ -----------------------------
RUNS = [dict(name="R1", R=410.0, afc="r1.csv", pfc="fchh410.csv", bin="410.bin"),
        dict(name="R2", R=2040.0, afc="r2.csv", pfc="fchh2040.csv", bin="2040.bin")]
L_NOM, C_NOM = 0.100, 6.0e-9           # Гн, Ф (по заданию)
# Измерено LCR-метром GW Instek LCR-7819 (последовательная схема, 1 В): L = 99.949 мГн (50 Гц),
# 99.922 мГн (500 Гц), 99.961 мГн (1.5 кГц). R_L прибор не показал (режим L/Q, «Q OVER»).
L_LCR = 99.944e-3                      # Гн, среднее по трём частотам (разброс ±0.02 мГн)
L_LCR_SIG = 0.1e-3                     # Гн, принятая погрешность (≈0.1 % - ОЦЕНКА по классу прибора)
SWEEP = (2990.0, 10990.0, 10.0)        # Start, Stop, Time (генератор)
PHASE_HINT = 6000.0                    # ориентир зоны для ФЧХ (см. analyze_rlc)
R_BOX_REL = 0.01                       # отн. погрешность магазина R - ОЦЕНКА (уточнить по паспорту МСР-60)
L_REL = L_LCR_SIG / L_LCR              # отн. погрешность L (по LCR-метру)
C_REL = 0.01                           # отн. погрешность C+C0 - ОЦЕНКА
RL_SIG = 5.0                           # Ом: неопределённость R_L (разброс оценок 51 Ом (АЧХ/ФЧХ) и 60 Ом (затухание)); измерить LCR-метром
SQ2 = np.sqrt(2.0)
COLORS = ["C0", "C1"]


# ---------------------------------- МОДЕЛЬ ----------------------------------
def H_model(f, L, C, C1, Rs):
    w = 2 * np.pi * np.asarray(f, float)
    Zt = (Rs + 1j * w * L) / (1 - w ** 2 * L * C + 1j * w * Rs * C)
    Y = 1j * w * C1 * Zt
    return Y / (1 + Y)


def U_model(f, p, Rbox):
    Ug, L, C, C1, RL = p
    return Ug * np.abs(H_model(f, L, C, C1, Rbox + RL))


def phi_model(f, p, Rbox):
    """Δφ = φ_ген - φ_U_C, на интервале (-π, 0] (как в методичке, рис. 5)."""
    Ug, L, C, C1, RL = p
    ph = -np.angle(H_model(f, L, C, C1, Rbox + RL))
    return np.where(ph > 0.3, ph - 2 * np.pi, ph)


# ------------------------------- ДАННЫЕ / ФИТ -------------------------------
def load_runs(ddir):
    data = []
    for r in RUNS:
        a = rlc.analyze_afc(os.path.join(ddir, r["afc"]), SWEEP)
        ph = rlc.analyze_phase(os.path.join(ddir, r["pfc"]), PHASE_HINT)
        d = dict(r)
        d.update(afc=a, pfc=ph)
        data.append(d)
    return data


def decimate(data, step=5):
    A, P = [], []
    for d in data:
        a, ph = d["afc"], d["pfc"]
        ok = np.isfinite(a["U"]) & (a["U"] > 0.2)
        A.append((a["f"][ok][::step], a["U"][ok][::step]))
        ok = np.isfinite(ph["dphi"])
        P.append((ph["f"][ok][::step], np.radians(ph["dphi"][ok])[::step]))
    return A, P


def fit_model(A, P, Rbox, p0, wA, wP, max_nfev=200):
    def res(p):
        out = []
        for i in range(len(A)):
            f, U = A[i]
            out.append((U_model(f, p, Rbox[i]) - U) / U * wA[i])
            f, ph = P[i]
            out.append((phi_model(f, p, Rbox[i]) - ph) * wP[i])
        out.append(np.array([(p[1] - L_LCR) / L_LCR_SIG]))      # L известна из измерения LCR-метром
        return np.concatenate(out)
    s = least_squares(res, p0, loss="soft_l1", f_scale=1.0, max_nfev=max_nfev,
                      x_scale=[1.0, 0.01, 1e-9, 1e-9, 10.0],
                      bounds=([0.1, 0.05, 3e-9, 1e-12, 0.0], [100, 0.2, 12e-9, 1e-7, 300]))
    return s.x


def robust_scales(p, A, P, Rbox):
    sA, sP = [], []
    for i in range(len(A)):
        f, U = A[i]
        r = (U_model(f, p, Rbox[i]) - U) / U
        sA.append(1.4826 * np.median(np.abs(r - np.median(r))))
        f, ph = P[i]
        r = phi_model(f, p, Rbox[i]) - ph
        sP.append(1.4826 * np.median(np.abs(r - np.median(r))))
    return np.array(sA), np.array(sP)


def block_resample(rng, arr_f, arr_y, nb=40):
    n = len(arr_f)
    nblk = max(n // nb, 1)
    starts = rng.integers(0, max(n - nb, 1), size=nblk)
    idx = np.concatenate([np.arange(s, min(s + nb, n)) for s in starts])
    return arr_f[idx], arr_y[idx]


# ------------------------- ОПЕРАЦИИ ИЗ МЕТОДИЧКИ ----------------------------
def _cross(x, y, level, i_start, direction):
    """Первая точка пересечения y=level от индекса i_start в направлении direction (+1/-1), линейная интерп."""
    i = i_start
    while 0 <= i + direction < len(y):
        y0, y1 = y[i] - level, y[i + direction] - level
        if np.isfinite(y0) and np.isfinite(y1) and y0 * y1 <= 0 and y0 != y1:
            t = y0 / (y0 - y1)
            return x[i] + t * (x[i + direction] - x[i])
        i += direction
    return np.nan


def lab_afc(f, U):
    """ν0 (максимум), U0, Δν на уровне U0/√2. Возвращает (ν0, U0, ν_лев, ν_прав)."""
    sel = (f > 3500) & (f < 9500) & np.isfinite(U)
    i = int(np.argmax(np.where(sel, U, -np.inf)))
    lvl = U[i] / SQ2
    return f[i], U[i], _cross(f, U, lvl, i, -1), _cross(f, U, lvl, i, +1)


def lab_pfc(f, ph):
    """ν(-π/2)=ν0, ν(-π/4)=ν+ (выше ν0), ν(-3π/4)=ν- (ниже ν0); Q = ν0/(ν+ - ν-) (зеркальное отражение нижней ветви).
    Учитываются только f > 1500 Гц: на очень низких частотах фаза контура возвращается к -π/2."""
    ok = np.isfinite(ph) & (f > 1500)
    g, y = f[ok], ph[ok]

    def cross(level):
        k = np.where((y[:-1] - level) * (y[1:] - level) <= 0)[0]
        return np.array([g[j] + (level - y[j]) / (y[j + 1] - y[j]) * (g[j + 1] - g[j])
                         for j in k if y[j] != y[j + 1]])
    c0 = cross(-np.pi / 2)
    if len(c0) == 0:
        return np.nan, np.nan, np.nan
    g0 = float(np.median(c0))
    cp, cm = cross(-np.pi / 4), cross(-3 * np.pi / 4)
    cp, cm = cp[cp > g0], cm[cm < g0]
    return g0, (float(cp.min()) if len(cp) else np.nan), (float(cm.max()) if len(cm) else np.nan)


def Q_from(f0, fl, fr):
    return f0 / (fr - fl) if np.isfinite(fl) and np.isfinite(fr) and fr > fl else np.nan


def direct_afc(f, U):
    """Прямое измерение на экспериментальной кривой (артефактные участки пропущены линейной интерполяцией)."""
    ok = np.isfinite(U)
    g = np.arange(f[ok].min(), f[ok].max(), 0.8)
    u = np.interp(g, f[ok], U[ok])
    u = savgol_filter(u, 201, 2)
    return lab_afc(g, u)


def direct_pfc(f, ph):
    ok = np.isfinite(ph)
    g = np.arange(f[ok].min(), f[ok].max(), 0.8)
    y = savgol_filter(np.interp(g, f[ok], ph[ok]), 401, 2)
    return lab_pfc(g, y)


def prop_sigma(func, vals, sigs):
    """Линейное распространение погрешностей (конечные разности) для Q = func(*vals)."""
    base = func(*vals)
    var = 0.0
    for k in range(len(vals)):
        if not np.isfinite(sigs[k]) or sigs[k] == 0 or not np.isfinite(vals[k]):
            continue
        v = list(vals)
        v[k] = v[k] + sigs[k]
        d = func(*v) - base
        if np.isfinite(d):
            var += d ** 2
    return np.sqrt(var)


def q_pfc(g0, gp, gm):
    """Q по методичке; если нижний уровень -3π/4 недостижим (у R2 фаза не опускается ниже ~-1.95 рад), -
    оценка по одному уровню -π/4 и соотношению идеального контура ν-·ν+ = ν0². Возвращает (Q, is_estimate)."""
    if np.isfinite(gm) and np.isfinite(gp) and np.isfinite(g0):
        return Q_from(g0, gm, gp), False
    if np.isfinite(gp) and np.isfinite(g0):
        return geo_Q_pfc(g0, gp), True
    return np.nan, True


# ---------------- п. 2.6: цуги (нарастание / затухание) ----------------
# Ручные курсорные измерения (лист «3. Установление и затухание» в 325.xlsx).
# Нарастание: (k, U_k, k+n, U_{k+n});  затухание: (m, U_m, m+n, U_{m+n}).
SIGMA_U_BURST = 0.05          # В, погрешность курсорного измерения
BURST = {
    "R1": {"U0": 8.24,
           "rise":  [(2, 2.84, 3, 4.48), (4, 5.68, 6, 7.12)],
           "decay": [(1, 5.52, 2, 3.80), (3, 2.56, 4, 1.72)]},
    "R2": {"U0": 2.09,
           "rise":  [(1, 0.73, 2, 1.80), (1, 0.73, 3, 2.04)],
           "decay": [(1, 1.82, 2, 0.32), (1, 1.82, 3, 0.052)]},
}


def burst_theta(U0, pairs, mode, sU=SIGMA_U_BURST):
    """Θ по парам амплитуд и их взвешенное среднее.
    rise : Θ = (1/n) ln[(U0-U_k)/(U0-U_{k+n})]  (U0 тоже с погрешностью sU)
    decay: Θ = (1/n) ln(U_m/U_{m+n})"""
    th, sth = [], []
    for k, u1, k2, u2 in pairs:
        n = k2 - k
        if mode == "rise":
            a, b = U0 - u1, U0 - u2
            th.append(np.log(a / b) / n)
            sth.append(sU / n * np.sqrt(1 / a**2 + 1 / b**2 + (1 / b - 1 / a) ** 2))
        else:
            th.append(np.log(u1 / u2) / n)
            sth.append(sU / n * np.sqrt(1 / u1**2 + 1 / u2**2))
    th, sth = np.array(th), np.array(sth)
    w = 1 / sth**2
    m = np.sum(w * th) / np.sum(w)
    return th, sth, m, 1 / np.sqrt(np.sum(w))


def burst_Q(name):
    """(Q_нар, σ, Q_зат, σ) для R1/R2; NaN, если данных нет."""
    if name not in BURST:
        return (np.nan,) * 4
    d = BURST[name]
    out = []
    for mode in ("rise", "decay"):
        _, _, m, s = burst_theta(d["U0"], d[mode], mode)
        out += [np.pi / m, np.pi * s / m**2]
    return tuple(out)


def geo_Q_pfc(f0, fp):
    """Идеальный контур 2-го порядка: ν-·ν+ = ν0² -> Δν = ν+ - ν0²/ν+ (для случая, когда нижняя ветвь не измерена)."""
    return f0 / (fp - f0 ** 2 / fp)


# ------------------------------------ MAIN ----------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DATA_DIR)
    ap.add_argument("--out", default=os.path.join(FIGURES_DIR, "q_analysis"))
    ap.add_argument("--nboot", type=int, default=120)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()
    if args.no_show:
        matplotlib.use("Agg")

    print("Загрузка АЧХ/ФЧХ ...")
    data = load_runs(args.dir)
    Rbox = [d["R"] for d in data]
    A, P = decimate(data)

    # ---------------- совместный фит ----------------
    p0 = np.array([7.3, L_LCR, 6.1e-9, 0.98e-9, 51.0])
    wA = np.full(len(data), 1 / 0.03)
    wP = np.full(len(data), 1 / 0.03)
    p = fit_model(A, P, Rbox, p0, wA, wP)
    sA, sP = robust_scales(p, A, P, Rbox)
    p = fit_model(A, P, Rbox, p, 1 / sA, 1 / sP)
    sA, sP = robust_scales(p, A, P, Rbox)
    Ug, L, C, C1, RL = p
    C_tot = C + C1
    f_free = 1 / (2 * np.pi * np.sqrt(L * C_tot))

    # ---------------- bootstrap (блочный) ----------------
    rng = np.random.default_rng(args.seed)
    boots, q_boot = [], []
    fgrid = np.arange(100.0, 14000.0, 1.0)

    def all_Q(pp):
        out = {}
        for i, d in enumerate(data):
            U = U_model(fgrid, pp, Rbox[i]); ph = phi_model(fgrid, pp, Rbox[i])
            f0, U0, fl, fr = lab_afc(fgrid, U)
            g0, gp, gm = lab_pfc(fgrid, ph)
            out[i] = (f0, Q_from(f0, fl, fr), g0, q_pfc(g0, gp, gm)[0])
        return out

    print(f"Bootstrap ({args.nboot}) ...")
    for _ in range(args.nboot):
        Ab, Pb = [], []
        for i in range(len(data)):
            Ab.append(block_resample(rng, *A[i]))
            Pb.append(block_resample(rng, *P[i]))
        try:
            pb = fit_model(Ab, Pb, Rbox, p, 1 / sA, 1 / sP, max_nfev=60)
            boots.append(pb)
            q_boot.append(all_Q(pb))
        except Exception:
            pass
    boots = np.array(boots)
    sp = boots.std(axis=0, ddof=1)
    qb = {i: np.array([qq[i] for qq in q_boot]) for i in range(len(data))}

    # ---------------- лабораторные величины ----------------
    spiral = {}
    for d in data:
        path = os.path.join(args.dir, d["bin"])
        try:
            spiral[d["name"]] = spi.analyze_file(path)
        except Exception as e:                       # noqa: BLE001
            spiral[d["name"]] = None
            print(f"[!] {d['bin']}: {e}", file=sys.stderr)

    rows = []
    curves = {}
    Q_model_ref = all_Q(p)
    for i, d in enumerate(data):
        R = d["R"]
        a, ph = d["afc"], d["pfc"]
        U = U_model(fgrid, p, R); phm = phi_model(fgrid, p, R)
        f0, U0, fl, fr = lab_afc(fgrid, U)
        Qa = Q_from(f0, fl, fr)
        g0, gp, gm = lab_pfc(fgrid, phm)
        Qp, est = q_pfc(g0, gp, gm)
        # прямые измерения по данным
        df0, dU0, dfl, dfr = direct_afc(a["f"], a["U"])
        Qa_d = Q_from(df0, dfl, dfr)
        dg0, dgp, dgm = direct_pfc(ph["f"], np.radians(ph["dphi"]))
        Qp_d, _ = q_pfc(dg0, dgp, dgm)
        sQa_boot = np.nanstd(qb[i][:, 1], ddof=1)
        # шум измерения -> погрешность частот пересечений (наклон кривой в точке пересечения)
        dU = np.gradient(U, fgrid)
        lvl = U0 / SQ2
        sig_l = sA[i] * lvl * SQ2 / max(abs(np.interp(fl, fgrid, dU)), 1e-9)
        sig_r = sA[i] * lvl * SQ2 / max(abs(np.interp(fr, fgrid, dU)), 1e-9)
        sQa_noise = prop_sigma(lambda x0, xl, xr: Q_from(x0, xl, xr), (f0, fl, fr), (0.0, sig_l, sig_r))
        sQa = np.sqrt(sQa_boot ** 2 + sQa_noise ** 2 + (abs(Qa - Qa_d) / 2 if np.isfinite(Qa_d) else 0.0) ** 2)
        dph = np.gradient(phm, fgrid)
        sg = [sP[i] / max(abs(np.interp(x, fgrid, dph)), 1e-9) if np.isfinite(x) else 0.0 for x in (g0, gp, gm)]
        sQp_noise = prop_sigma(lambda a_, b_, c_: q_pfc(a_, b_, c_)[0], (g0, gp, gm), tuple(sg))
        sQp_boot = np.nanstd(qb[i][:, 3], ddof=1)
        sQp_boot = np.hypot(sQp_boot, sQp_noise)
        if not est and np.isfinite(Qp_d):
            sQp = np.hypot(sQp_boot, abs(Qp - Qp_d) / 2)
            p_note = "уровни -π/4, -π/2, -3π/4 достигнуты"
        elif not est:
            sQp = sQp_boot
            p_note = "по модельной кривой"
        else:
            sQp = None                                   # заполняется ниже, после R1 (калибровка оценки)
            p_note = "уровень -3π/4 недостижим: оценка по -π/4 и ν-·ν+ = ν0²"
        # свободные колебания
        sp_ = spiral[d["name"]]
        if sp_ is not None:
            Q_th, sQ_th = np.pi / sp_["theta"], np.pi * sp_["s_theta"] / sp_["theta"] ** 2
            s_ex = np.hypot(sp_["s_theta_ex"], (sp_["theta"] - sp_["theta_ex"]) / 2)
            Q_sp, sQ_sp = np.pi / sp_["theta_ex"], np.pi * s_ex / sp_["theta_ex"] ** 2
        else:
            Q_th = sQ_th = Q_sp = sQ_sp = np.nan
        # теория f(L,C,R)
        Rs = R + RL
        Q_theory = np.sqrt(L_LCR / C_tot) / Rs
        rel = np.sqrt((0.5 * L_REL) ** 2 + (0.5 * C_REL) ** 2 + (np.hypot(R_BOX_REL * R, RL_SIG) / Rs) ** 2)
        sQ_theory = Q_theory * rel
        Q_nom = np.sqrt(L_LCR / C_NOM) / (R + RL)
        rows.append(dict(name=d["name"], R=R, Rs=Rs, f0=f0, U0=U0, fl=fl, fr=fr, Qa=Qa, sQa=sQa,
                         Qa_d=Qa_d, df0=df0, dfl=dfl, dfr=dfr,
                         g0=g0, gp=gp, gm=gm, Qp=Qp, sQp=sQp, Qp_est=est, sQp_boot=sQp_boot, Qp_d=Qp_d, dg0=dg0, dgp=dgp, dgm=dgm, p_note=p_note,
                         Q_th=Q_th, sQ_th=sQ_th, Q_sp=Q_sp, sQ_sp=sQ_sp,
                         Q_theory=Q_theory, sQ_theory=sQ_theory, Q_nom=Q_nom))
        curves[i] = (U, phm)

    # для оценок «по одному уровню» (R2): поправка = отличие такой оценки от полного метода на R1
    full = [r for r in rows if not r["Qp_est"]]
    if full:
        r0 = full[0]
        Qgeo0 = geo_Q_pfc(r0["g0"], r0["gp"])
        calib = abs(Qgeo0 - r0["Qp"]) / r0["Qp"]
    else:
        calib = 0.2
    for r in rows:
        if r["Qp_est"]:
            r["sQp"] = r["Qp"] * np.hypot(calib, r["sQp_boot"] / r["Qp"])
    # ---------------- вывод ----------------
    print("\n=== Параметры модели (совместная аппроксимация 2 АЧХ + 2 ФЧХ) ===")
    names = ["U_g, В", "L, мГн", "C, нФ", "C1, нФ", "R_L, Ом"]
    scale = [1, 1e3, 1e9, 1e9, 1]
    for nme, v, s_, k in zip(names, p, sp, scale):
        print(f"  {nme:8s} = {v*k:9.3f} ± {s_*k:.3f}")
    print(f"  остаток аппроксимации: АЧХ {100*sA[0]:.1f} % (R1), {100*sA[1]:.1f} % (R2); "
          f"ФЧХ {sP[0]:.3f} рад (R1), {sP[1]:.3f} рад (R2)")
    print(f"  C+C1 = {C_tot*1e9:.2f} нФ -> частота свободных колебаний 1/(2π√(L(C+C1))) = {f_free:.0f} Гц "
          f"(по осциллограммам спирали: f0 = {np.mean([np.hypot(2*np.pi*spiral[k]['f_d'], spiral[k]['beta'])/2/np.pi for k in spiral if spiral[k]]):.0f} Гц)")

    print("\n=== АЧХ, п. 2.7.7-8 (ширина на уровне 1/√2) ===")
    for r in rows:
        print(f"{r['name']} (R={r['R']:.0f}): ν0={r['f0']:.0f} Гц, U0={r['U0']:.2f} В (модель); "
              f"ν_лев={r['fl']:.0f}, ν_прав={r['fr']:.0f}, Δν={r['fr']-r['fl']:.0f} Гц -> Q = {r['Qa']:.2f} ± {r['sQa']:.2f}   "
              f"[прямое измерение по точкам: ν0={r['df0']:.0f}, Δν={r['dfr']-r['dfl']:.0f}, Q={r['Qa_d']:.2f}]")

    print("\n=== ФЧХ, п. 2.7.9-10 (зеркальное отражение, Δν на уровне -π/4) ===")
    for r in rows:
        print(f"{r['name']} (R={r['R']:.0f}): ν(-π/2)={r['g0']:.0f}, ν(-π/4)={r['gp']:.0f}, ν(-3π/4)=" + (f"{r['gm']:.0f}" if np.isfinite(r['gm']) else "недостижим") + " Гц (модель) "
              f"-> " + (f"Δν={r['gp']-r['gm']:.0f}, " if not r['Qp_est'] else "") + f"Q = {r['Qp']:.2f} ± {r['sQp']:.2f}" + (" (оценка †)" if r['Qp_est'] else "") + "   "
              f"[по точкам: ν(-π/2)={r['dg0']:.0f}, ν(-π/4)={r['dgp']:.0f}, ν(-3π/4)="
              + (f"{r['dgm']:.0f}" if np.isfinite(r['dgm']) else "не достигнут") + (f", Q={r['Qp_d']:.2f}" if np.isfinite(r['Qp_d']) else "") + f"]  ({r['p_note']})")

    # ---------------- график 1: п. 2.7.7, 2.7.9-10 ----------------
    fig, ax = plt.subplots(2, 1, figsize=(11, 12))
    a = ax[0]
    for i, (d, r) in enumerate(zip(data, rows)):
        c = COLORS[i]
        af = d["afc"]
        a.plot(af["f"] / r["f0"], af["U"] / r["U0"], lw=0.9, color=c, alpha=0.9,
               label=f"{r['name']} = {r['R']:.0f} Ом: эксперимент")
        a.plot(fgrid / r["f0"], curves[i][0] / r["U0"], "--", color="k", lw=1.1,
               label="аппроксимация моделью контура" if i == 0 else None)
        a.plot([r["fl"] / r["f0"], r["fr"] / r["f0"]], [1 / SQ2] * 2, "o", color=c, mec="k", ms=7, zorder=5,
               label=f"Δν = {r['fr']-r['fl']:.0f} Гц, Q = ν0/Δν = {r['Qa']:.2f} ± {r['sQa']:.2f}")
    a.axhline(1 / SQ2, color="gray", lw=0.8, ls="--")
    a.text(0.53, 1 / SQ2 + 0.015, "уровень $1/\\sqrt{2}$", color="gray", fontsize=10)
    a.axvline(1.0, color="gray", lw=0.6, ls=":")
    a.set_xlim(0.5, 1.8); a.set_ylim(0, 1.08)
    a.set_xlabel("$\\nu/\\nu_0$"); a.set_ylabel("$U/U_0$")
    a.set_title("Резонансные кривые $U/U_0 = f(\\nu/\\nu_0)$ (п. 2.7.7-8)")
    a.legend(loc="upper right", fontsize=10); a.grid(True, ls="--", alpha=0.5)

    a = ax[1]
    for i, (d, r) in enumerate(zip(data, rows)):
        c = COLORS[i]
        ph = d["pfc"]
        fpt, ypt = ph["f"], np.radians(ph["dphi"])
        a.plot(fpt, ypt, ".", ms=1.6, color=c, label=f"{r['name']} = {r['R']:.0f} Ом: эксперимент")
        a.plot(fgrid, curves[i][1], "-", color="k", lw=0.9, alpha=0.7,
               label="аппроксимация моделью" if i == 0 else None)
        low = np.isfinite(ypt) & (fpt < r["g0"])
        a.plot(fpt[low], -np.pi - ypt[low], ".", ms=1.6, color=c, alpha=0.45)
        m = (fgrid < r["g0"])
        a.plot(fgrid[m], -np.pi - curves[i][1][m], "--", color=c, lw=1.0)
        if np.isfinite(r["gp"]) and np.isfinite(r["gm"]):
            a.annotate("", xy=(r["gp"], -np.pi / 4), xytext=(r["gm"], -np.pi / 4),
                       arrowprops=dict(arrowstyle="<->", color=c, lw=2.0))
        a.plot([r["g0"]], [-np.pi / 2], "o", color=c, mec="k", ms=7, zorder=6)
        if np.isfinite(r["gp"]) and np.isfinite(r["gm"]):
            a.text(0.5 * (r["gp"] + r["gm"]), -np.pi / 4 + 0.1,
                   f"Δν={r['gp']-r['gm']:.0f} Гц\nQ={r['Qp']:.2f}±{r['sQp']:.2f}", color=c, ha="center", fontsize=10.5)
        elif np.isfinite(r["gp"]):
            a.text(r["gp"] + 150, -np.pi / 4 - 0.32,
                   f"Q≈{r['Qp']:.2f}±{r['sQp']:.2f} †\n(нижний уровень −3π/4\nне достигается)", color=c, ha="left", fontsize=10)
    a.axhline(-np.pi / 2, color="gray", lw=0.8); a.axhline(-np.pi / 4, color="gray", lw=0.8, ls="--")
    a.set_yticks(np.pi * np.array([-1, -0.75, -0.5, -0.25, 0]))
    a.set_yticklabels(["−π", "−3π/4", "−π/2", "−π/4", "0"])
    a.set_ylim(-np.pi * 1.03, 0.1); a.set_xlim(2500, 11000)
    a.set_xlabel("$\\nu$, Гц"); a.set_ylabel("$\\Delta\\varphi$, рад")
    a.set_title("ФЧХ и зеркальное отражение нижней ветви (пунктир) относительно $-\\pi/2$ (п. 2.7.9-10)")
    a.legend(loc="lower right", fontsize=10); a.grid(True, ls="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(args.out + "_curves.pdf")

    # ---------------- таблица (п. 13) ----------------
    hdr = ["R", "f(L,C,R)", "f(Θ)", "Спираль", "АЧХ", "ФЧХ", "Нарастание", "Затухание"]

    def cell(v, s):
        return f"{v:.2f} ± {s:.2f}" if np.isfinite(v) else "—"
    table = []
    for r in rows:
        table.append([f"{r['name']} = {r['R']:.0f} Ом",
                      cell(r["Q_theory"], r["sQ_theory"]), cell(r["Q_th"], r["sQ_th"]), cell(r["Q_sp"], r["sQ_sp"]),
                      cell(r["Qa"], r["sQa"]), cell(r["Qp"], r["sQp"]) + (" †" if r["Qp_est"] else ""),
                      cell(*burst_Q(r["name"])[:2]), cell(*burst_Q(r["name"])[2:])])
    print("\n=== Сводная таблица добротности Q (п. 2.7.13), погрешности 1σ ===")
    w = [max(len(hdr[j]), max(len(t[j]) for t in table)) + 2 for j in range(len(hdr))]
    print("".join(h.center(w[j]) for j, h in enumerate(hdr)))
    print("-" * sum(w))
    for t in table:
        print("".join(c.center(w[j]) for j, c in enumerate(t)))
    print("\nЦуги (п. 2.6), σ_U = %.2f В:" % SIGMA_U_BURST)
    for nm, d in BURST.items():
        for mode, lab in (("rise", "нарастание"), ("decay", "затухание")):
            th, sth, m, s = burst_theta(d["U0"], d[mode], mode)
            print(f"  {nm} {lab:11s}: Θ = " + ", ".join(f"{t:.3f}±{e:.3f}" for t, e in zip(th, sth))
                  + f"  -> <Θ> = {m:.3f} ± {s:.3f},  Q = {np.pi/m:.2f} ± {np.pi*s/m**2:.2f}")
    print("† - оценка: у R2 фаза не достигает уровня -3π/4 (минимум около -1.95 рад), поэтому Δν найдено по уровню -π/4 "
          f"и соотношению ν-·ν+ = ν0² (на R1 такая оценка отличается от полного метода на {100*calib:.0f} %, эта величина заложена в погрешность).")
    print(f"f(L,C,R): Q = √(L/(C+C0))/(R+R_L), L = {L_LCR*1e3:.2f} мГн (LCR-метр), C+C0 = "
          f"{C_tot*1e9:.2f} нФ, R_L = {RL:.1f} ± {RL_SIG:.0f} Ом (из аппроксимации; разброс оценок 51 и 60 Ом).")
    print("R_L: LCR-метр в режиме L/Q показал «Q OVER» (R не отображается); значение R_L взято из аппроксимации АЧХ/ФЧХ "
          "и должно быть заменено измерением в режиме L/R.")
    print("АЧХ, R2: при Q≲2 и несимметричной кривой (контур возбуждается через C1) ширина на уровне 1/√2 занижает Q "
          "(метод рассчитан на Q≫1); это методическая, а не случайная погрешность.")
    print("Если брать C = 6.0 нФ без C0 (номинал по магазину): "
          + "; ".join(f"{r['name']}: Q = {r['Q_nom']:.2f}" for r in rows) + ".")

    import csv
    with open(args.out + "_table.csv", "w", newline="", encoding="utf-8-sig") as fh:
        wtr = csv.writer(fh, delimiter=";")
        wtr.writerow(hdr)
        wtr.writerows(table)

    fig2, a2 = plt.subplots(figsize=(13, 2.4 + 0.5 * len(table)))
    a2.axis("off")
    tb = a2.table(cellText=table, colLabels=hdr, loc="center", cellLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(12); tb.scale(1, 2.0)
    a2.set_title("Добротность контура Q, определённая разными способами (±1σ)", pad=14)
    fig2.text(0.5, 0.03, "Нарастание/затухание: Q = π/⟨Θ⟩ по цугам (п. 2.6);  † — оценка: у R2 фаза не достигает уровня −3π/4", ha="center", fontsize=10)
    fig2.savefig(args.out + "_table.pdf", bbox_inches="tight")
    print(f"\nСохранено: {args.out}_curves.pdf, {args.out}_table.(pdf|csv)")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
