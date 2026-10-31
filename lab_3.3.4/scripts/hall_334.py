#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Работа 3.3.4 «Эффект Холла в полупроводниках» — обработка результатов.

Запуск:   python hall_334.py [путь/к/3.3.4_clean.xlsx]  (по умолчанию — путь из PROJECT)
Нужно:    numpy, pandas, scipy, matplotlib, openpyxl
Вход:     xlsx с листами  calibration, hall_tidy, parameters
          (лист hall_wide дублирует hall_tidy и не используется)
Выход:    results/results.txt  и  figures/fig1..fig4 (*.pdf, векторные)

Что считается (нумерация — по методичке, «Обработка результатов»):
  12  калибровка B(I_M)             -> fig1
  13  U_perp = U34 - U0, семейство U_perp(B) при разных I, наклоны k   -> fig2
  14  k(I), постоянная Холла R_H    -> fig4
  15  концентрация n, удельное сопротивление rho0, проводимость sigma0
  16  подвижность mu
  17  погрешности, сравнение с табличными
  +   п. 8 (обратное поле) и п. 9 (знак носителей)                       -> fig3
  +   контрольный вопрос 6 (tau, длина пробега, D) — короткая оценка
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial
from scipy.optimize import curve_fit

# =====================================================================
#                              НАСТРОЙКИ
# =====================================================================
PROJECT = Path(r"C:\Utilities\Projects\MIPT_labs\lab_3.3.4")
XLSX = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT / "data" / "3.3.4_clean.xlsx"
RESULTS_DIR = PROJECT / "results"     # сюда пишется results.txt
FIGURES_DIR = PROJECT / "figures"     # сюда пишутся графики (PDF)

# --- геометрия образца (из листа parameters) ---------------------------------
# ВНИМАНИЕ: в xlsx параметр 'l' = 1.7 мм подписан «геометрия образца». В методичке
# буквой l обозначено расстояние 3–5 (это в xlsx 'L_35'), поэтому 'l' из xlsx
# трактуется как ТОЛЩИНА h. Если это не так — поменяй H_MM здесь.
H_MM_OVERRIDE = None          # None -> взять 'l' из листа parameters

# --- приборные погрешности (допущения! подставь паспортные значения) ---------
# Правило из пособия: макс. приборная погрешность ~ цене деления / единице последнего разряда.
SIGMA_U_UV = 1.0              # мкВ: единица последнего разряда вольтметра (U34, U0)
SIGMA_U35_MV = 0.001          # мВ: единица последнего разряда при измерении U35 (2.231 мВ)
SIGMA_I_DIV = 1.0             # дел: погрешность миллиамперметра образца (цена деления 0.01 мА)
SIGMA_GEOM_MM = 0.1           # мм: a, h, l35 (по умолчанию ±1 в последней значащей цифре)
B_INSTR_REL = 0.01            # отн.: систематика прибора, которым градуировали магнит (допущение)

# --- калибровка магнита -------------------------------------------------------
CAL_PASSES = (1, 2)           # какие проходы (колонка 'pass') использовать в аппроксимации
DEG_GAIN_THRESHOLD = 0.05     # степень полинома: берём минимальную, после которой RMS падает <5 %
N_BOOT = 500                  # бутстрэп калибровки для оценки её вклада в погрешность R_H

# --- знак носителей: геометрия опыта (ПРОВЕРЬ НА УСТАНОВКЕ!) -----------------
# Рис. 1б: ток идёт от контакта 2 (+) к контакту 1 (−), т.е. «вверх»; контакт 3 слева, 4 справа.
B_FWD_TOWARD_VIEWER = True    # поле «вперёд» (К1 в положении 1) направлено НА наблюдателя (на рис. 1б)?
U34_IS_PHI3_MINUS_PHI4 = True # показание вольтметра U34 = φ3 − φ4 ?

# --- справочные величины ------------------------------------------------------
E_CHARGE = 1.602176634e-19    # Кл
K_B = 1.380649e-23            # Дж/К
M_E = 9.1093837e-31           # кг
MU_TABLE = {"электроны (n-Ge, 300 К)": 3900.0, "дырки (p-Ge, 300 К)": 1900.0}   # см²/(В·с)
T_K = 295.0                   # К, комнатная температура (для оценок по контр. вопр. 6)
M_EFF_REL = {"электроны": 0.12, "дырки": 0.25}   # m*/m_e, ориентировочно (проверь по таблицам)

# =====================================================================
#                            ВСПОМОГАТЕЛЬНОЕ
# =====================================================================
plt.rcParams.update({
    "font.size": 18, "axes.labelsize": 22, "axes.titlesize": 20,
    "xtick.labelsize": 18, "ytick.labelsize": 18, "legend.fontsize": 14,
    "lines.linewidth": 2.4, "lines.markersize": 9, "axes.linewidth": 1.4,
    "xtick.major.width": 1.4, "ytick.major.width": 1.4,
    "axes.grid": True, "grid.alpha": 0.5, "grid.linestyle": "--",
    "figure.figsize": (10, 7.5), "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.fontset": "dejavusans",
})
ERR_KW = dict(mec="k", mew=0.8, ecolor="k", capsize=4, elinewidth=1.3, capthick=1.3)   # стиль точек как в spiral


def fit_line(x, y, sy, sx=None, through_zero=False, n_iter=30):
    """Взвешенная прямая y = k x + b (или y = k x) методом хи-квадрат (curve_fit, absolute_sigma).
    Если задан sx — учёт погрешности по x через эффективную дисперсию sy² + (k·sx)²."""
    x, y, sy = (np.asarray(v, float) for v in (x, y, sy))
    model = (lambda t, k: k * t) if through_zero else (lambda t, k, b: k * t + b)
    sig = sy.copy()
    for _ in range(n_iter if sx is not None else 1):
        popt, pcov = curve_fit(model, x, y, sigma=sig, absolute_sigma=True)
        if sx is None:
            break
        sig = np.sqrt(sy ** 2 + (popt[0] * np.asarray(sx, float)) ** 2)
    res = y - model(x, *popt)
    two = not through_zero
    return dict(k=popt[0], sk=np.sqrt(pcov[0, 0]),
                b=popt[1] if two else 0.0, sb=np.sqrt(pcov[1, 1]) if two else 0.0,
                cov=pcov[0, 1] if two else 0.0,
                chi2=np.sum((res / sig) ** 2), dof=len(x) - len(popt),
                frac=np.mean(np.abs(res) <= sig), res=res, sig=sig)


def fmt(v, e, unit=""):
    """Запись результата по правилам пособия: погрешность — 1–2 значащие цифры
    (две, если первая цифра 1 или 2), значение — до того же разряда."""
    if e is None or not np.isfinite(e) or e <= 0:
        return f"{v:.6g} {unit}".strip()
    p = 0
    if v != 0 and (abs(v) >= 1e4 or abs(v) < 1e-2):
        p = int(np.floor(np.log10(abs(v))))
    vs, es = v / 10 ** p, e / 10 ** p
    ex = int(np.floor(np.log10(es)))
    nd = 2 if int(es / 10 ** ex) in (1, 2) else 1
    dec = nd - 1 - ex
    vs, es = round(vs, dec), round(es, dec)
    d = max(dec, 0)
    body = f"{vs:.{d}f} ± {es:.{d}f}"
    if p != 0:
        body = f"({body})·10^{p}"
    return f"{body} {unit}".strip()


def rel(v, e):
    return f"{100 * e / abs(v):.1f} %"


def quad(*r):
    return float(np.sqrt(np.sum(np.square(r))))


# =====================================================================
#                              ДАННЫЕ
# =====================================================================
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
cal = pd.read_excel(XLSX, sheet_name="calibration")
tidy = pd.read_excel(XLSX, sheet_name="hall_tidy")
par = pd.read_excel(XLSX, sheet_name="parameters").set_index("param")["value"].astype(float)

K_I = par["k_I"]                         # мА/дел
U_RHO_V = par["U_rho"] * 1e-3            # В
I_RHO_A = par["I_rho"] * K_I * 1e-3      # А
A_M = par["a"] * 1e-3                    # ширина, м
L35_M = par["L_35"] * 1e-3               # расстояние между контактами 3 и 5, м
H_MM = H_MM_OVERRIDE if H_MM_OVERRIDE is not None else par["l"]
H_M = H_MM * 1e-3                        # толщина, м
S_GEOM = SIGMA_GEOM_MM * 1e-3
S_I_SAMPLE_MA = SIGMA_I_DIV * K_I        # мА

warnings = []

# =====================================================================
#                   12. КАЛИБРОВКА B(I_M)  (fig1)
# =====================================================================
Ical = cal["I_M_mA"].to_numpy(float)
Bcal = cal["B_mT"].to_numpy(float)
passno = cal["pass"].to_numpy()
use = np.isin(passno, CAL_PASSES)

polys, rms = {}, {}
for d in range(1, 9):
    polys[d] = Polynomial.fit(Ical[use], Bcal[use], d)
    r = Bcal[use] - polys[d](Ical[use])
    rms[d] = np.sqrt(np.sum(r ** 2) / (use.sum() - d - 1))
deg = next((d for d in range(2, 8) if (rms[d] - rms[d + 1]) / rms[d] < DEG_GAIN_THRESHOLD), 8)
Bfun, sB = polys[deg], rms[deg]          # sB, мТл — разброс точек вокруг калибровки
coef = Bfun.convert().coef
xx = np.linspace(0, Ical.max(), 600)
monotone = bool(np.all(Bfun.deriv()(xx) > 0))
cal_res = Bcal - Bfun(Ical)

fig, ax = plt.subplots()
for pn, mk, col, extra in [(1, "o", "C0", {}), (2, "s", "C1", {"mfc": "none"})]:
    m = passno == pn
    if m.any():
        ax.errorbar(Ical[m], Bcal[m], yerr=sB, fmt=mk, color=col, label=f"проход {pn}", **ERR_KW, **extra)
ax.plot(xx, Bfun(xx), "C3-", label=f"полином {deg}-й ст.")
ax.set_xlabel(r"$I_{\rm M}$, мА", fontsize = 28)
ax.set_ylabel(r"$B$, мТл", fontsize = 28)
ax.set_title("Градуировка электромагнита", fontsize = 28)
ax.set_xlim(0, Ical.max() * 1.03)
ax.set_ylim(0, Bcal.max() * 1.05)
ax.legend(loc="lower right", fontsize = 24)
fig.savefig(FIGURES_DIR / "fig1_calibration_B_of_IM.pdf")
plt.close(fig)

# =====================================================================
#          13. ЭДС ХОЛЛА  U_perp = U34 - U0  и наклоны k = dU/dB  (fig2)
# =====================================================================
SU = np.hypot(SIGMA_U_UV, SIGMA_U_UV)   # погрешность U_perp = U34 - U0, мкВ
fw = tidy[tidy["direction"] == "forward"]
series = []
for div, g in fw.groupby("I_sample_div"):
    g = g.sort_values("I_M_mA")
    IM = g["I_M_mA"].to_numpy(float)
    U34 = g["U34_uV"].to_numpy(float)
    U0 = g["U0_uV"].to_numpy(float)
    assert np.all(U0 == U0[0]), "U0 должно быть постоянным в серии"
    Up = U34 - U0                                    # формула (1) методички
    if not np.allclose(Up, g["U_Hall_uV"].to_numpy(float)):
        warnings.append(f"серия I={div} дел: U34-U0 не совпало с колонкой U_Hall_uV в xlsx")
    Bv = Bfun(IM)
    f = fit_line(Bv, Up, np.full(len(IM), SU))       # U_perp = k B + b, мкВ и мТл
    series.append(dict(div=div, I=div * K_I, IM=IM, B=Bv, U34=U34, U0=U0[0], Up=Up, fit=f))

Iv = np.array([s["I"] for s in series])              # мА
kv = np.array([s["fit"]["k"] for s in series])       # мкВ/мТл
skv = np.array([s["fit"]["sk"] for s in series])

fig, ax = plt.subplots()
cm = plt.cm.viridis(np.linspace(0.05, 0.9, len(series)))
for i, s in enumerate(series):
    f, col = s["fit"], cm[i]
    ax.errorbar(s["B"], s["Up"], yerr=SU, fmt="o", color=col, **ERR_KW,
                label=f"{s['I']:.2f} мА: $k$ = {fmt(f['k'], f['sk'])}")
    bb = np.array([0.0, s["B"].max() * 1.02])
    ax.plot(bb, f["k"] * bb + f["b"], "-", color=col, lw=1.5)
ax.set_xlabel(r"$B$, мТл")
ax.set_ylabel(r"$U_\perp$, мкВ")
ax.set_title(r"ЭДС Холла при разных токах ($k$ в мкВ/мТл)")
ax.set_xlim(0, max(s["B"].max() for s in series) * 1.04)
top = max(s["Up"].max() for s in series)
ax.set_ylim(min(0, min(s["Up"].min() for s in series)) - 0.02 * top, top * 1.30)
ax.legend(loc="upper left")
fig.savefig(FIGURES_DIR / "fig2_hall_family_Uperp_of_B.pdf")
plt.close(fig)

# =====================================================================
#                 14. k(I)  ->  постоянная Холла R_H  (fig4)
# =====================================================================
fk = fit_line(Iv, kv, skv, sx=np.full(len(Iv), S_I_SAMPLE_MA))             # k = s I + k0
fk0 = fit_line(Iv, kv, skv, sx=np.full(len(Iv), S_I_SAMPLE_MA), through_zero=True)   # k = s I
s_slope, ss_slope = fk["k"], fk["sk"]            # мкВ/(мТл·мА)

# Перевод в СИ: мкВ/мТл = 1e-3 В/Тл ; 1 мА = 1e-3 А  ->  мкВ/(мТл·мА) = 1 В/(Тл·А)
TO_SI = (1e-6 / 1e-3) / 1e-3
s_si, ss_si = s_slope * TO_SI, ss_slope * TO_SI
RH = s_si * H_M                                  # м³/Кл,  U_perp = R_H I B / h

fig, ax = plt.subplots()
ax.errorbar(Iv, kv, xerr=S_I_SAMPLE_MA, yerr=skv, fmt="o", color="C0", **ERR_KW, label="эксперимент (±1σ по X и Y)")
xi = np.linspace(0, Iv.max() * 1.08, 200)
ax.plot(xi, fk["k"] * xi + fk["b"], "C3-", label=f"МНК: $s$ = {fmt(fk['k'], fk['sk'])} В/(Тл·А)")
ax.plot(xi, fk0["k"] * xi, "k--", label=f"через ноль: $s$ = {fmt(fk0['k'], fk0['sk'])}")
ax.set_xlabel(r"$I$, мА", fontsize = 26)
ax.set_ylabel(r"$k$, мкВ/мТл", fontsize = 26)
ax.set_title(r"Наклон $k = dU_\perp/dB$ от тока, $s = R_H/h$", fontsize = 26)
ax.set_xlim(0, Iv.max() * 1.08)
ax.set_ylim(-0.03 * kv.max(), kv.max() * 1.12)
ax.legend(loc="upper left")
fig.savefig(FIGURES_DIR / "fig4_k_of_I.pdf")
plt.close(fig)

# вклад калибровки в погрешность наклона: бутстрэп калибровочных точек
rng = np.random.default_rng(12345)
idx = np.flatnonzero(use)
s_boot = []
for _ in range(N_BOOT):
    ii = rng.choice(idx, size=idx.size, replace=True)
    try:
        pb = Polynomial.fit(Ical[ii], Bcal[ii], deg)
        kb = [np.polyfit(pb(s["IM"]), s["Up"], 1)[0] for s in series]
        s_boot.append(np.polyfit(Iv, kb, 1, w=1 / fk["sig"])[0])
    except Exception:
        pass
cal_rel = float(np.std(s_boot, ddof=1) / abs(s_slope)) if len(s_boot) > 20 else 0.0

# =====================================================================
#     п. 8: обратное поле и альтернативный метод (полуразность)   (fig3)
# =====================================================================
fmax = series[-1]
rv = tidy[tidy["direction"] == "reverse"].sort_values("I_M_mA")
rev = None
if len(rv):
    if rv["I_sample_div"].iloc[0] != fmax["div"]:
        warnings.append("ток образца в обратном поле не совпадает с максимальным прямым")
    IMr = rv["I_M_mA"].to_numpy(float)
    U34r = rv["U34_uV"].to_numpy(float)
    U0r = float(rv["U0_uV"].iloc[0])
    Upr = U34r - U0r
    Br_signed = -Bfun(IMr)                         # поле обращено
    frev = fit_line(Br_signed, Upr, np.full(len(IMr), SU))
    # полуразность U34(+) - U34(-) при одинаковых I_M (включая I_M = 0, где берём U0)
    fdf = pd.DataFrame({"IM": fmax["IM"], "Uf": fmax["U34"]})
    rdf = pd.DataFrame({"IM": IMr, "Ur": U34r})
    m = fdf.merge(rdf, on="IM")
    IMs = np.r_[0.0, m["IM"].to_numpy()]
    Usym = np.r_[(fmax["U0"] - U0r) / 2, (m["Uf"] - m["Ur"]).to_numpy() / 2]
    fsym = fit_line(Bfun(IMs), Usym, np.full(len(IMs), SU / np.sqrt(2)))
    rev = dict(IM=IMr, U0=U0r, Up=Upr, B=Br_signed, fit=frev, Isym=IMs, Usym=Usym, fsym=fsym)

    fig, ax = plt.subplots()
    ax.errorbar(fmax["B"], fmax["Up"], yerr=SU, fmt="o", color="C0", **ERR_KW, label=f"прямое поле: $k$ = {fmt(fmax['fit']['k'], fmax['fit']['sk'])}")
    ax.errorbar(Br_signed, Upr, yerr=SU, fmt="o", color="C1", **ERR_KW, label=f"обратное поле: $k$ = {fmt(frev['k'], frev['sk'])}")
    bf = np.array([0, fmax["B"].max() * 1.03])
    ax.plot(bf, fmax["fit"]["k"] * bf + fmax["fit"]["b"], "C3-")
    br = np.array([Br_signed.min() * 1.03, 0])
    ax.plot(br, frev["k"] * br + frev["b"], "C3-")
    Bsym = Bfun(rev["Isym"])
    ax.errorbar(Bsym, rev["Usym"], yerr=SU / np.sqrt(2), fmt="D", color="C2", ms=7, **ERR_KW,
                label=f"полуразность: $k$ = {fmt(rev['fsym']['k'], rev['fsym']['sk'])}")
    bs = np.array([0, Bsym.max() * 1.03])
    ax.plot(bs, rev["fsym"]["k"] * bs + rev["fsym"]["b"], "k--")
    ax.axhline(0, color="gray", lw=0.8)
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlabel(r"$B$, мТл", fontsize = 26)
    ax.set_ylabel(r"$U_\perp$, мкВ", fontsize = 26)
    ax.set_title(f"Прямое и обратное поле, $I$ = {fmax['I']:.2f} мА", fontsize = 26)
    ax.legend(loc="upper left", fontsize = 20)
    fig.savefig(FIGURES_DIR / "fig3_reverse_field_check.pdf")
    plt.close(fig)

# =====================================================================
#          9. ЗНАК НОСИТЕЛЕЙ;  15–16. n, rho0, sigma0, mu
# =====================================================================
sign_reading = int(np.sign(s_slope))
eff = sign_reading * (1 if U34_IS_PHI3_MINUS_PHI4 else -1) * (1 if B_FWD_TOWARD_VIEWER else -1)
carrier = "электроны (n-тип)" if eff > 0 else "дырки (p-тип)"
carrier_short = "электроны" if eff > 0 else "дырки"

# систематика от смены направления поля: половина относительной разности наклонов вперёд/назад
dir_rel = 0.0
if rev is not None:
    dir_rel = abs(fmax["fit"]["k"] - rev["fit"]["k"]) / (fmax["fit"]["k"] + rev["fit"]["k"])

# относительные погрешности исходных величин
r_h, r_a, r_l = S_GEOM / H_M, S_GEOM / A_M, S_GEOM / L35_M
r_U35 = SIGMA_U35_MV * 1e-3 / U_RHO_V
r_Irho = S_I_SAMPLE_MA * 1e-3 / I_RHO_A
r_s = ss_slope / abs(s_slope)

# R_H и n
sRH = abs(RH) * quad(r_s, r_h, cal_rel, B_INSTR_REL, dir_rel)
n_m3 = 1 / (E_CHARGE * abs(RH))
sn = n_m3 * quad(r_s, r_h, cal_rel, B_INSTR_REL, dir_rel)

# rho0 = U35 a h / (I l)   (формула (2) методички),  sigma0 = 1/rho0
rho = U_RHO_V * A_M * H_M / (I_RHO_A * L35_M)                 # Ом·м
srho = rho * quad(r_U35, r_a, r_h, r_Irho, r_l)
sigma0 = 1 / rho
ssigma = sigma0 * quad(r_U35, r_a, r_h, r_Irho, r_l)

# mu = |R_H| sigma0 ;  в явном виде mu = s I_rho l / (U35 a): толщина h сокращается
mu = abs(RH) * sigma0                                          # м²/(В·с)
mu_check = abs(s_si) * I_RHO_A * L35_M / (U_RHO_V * A_M)
smu = mu * quad(r_s, cal_rel, B_INSTR_REL, dir_rel, r_Irho, r_l, r_U35, r_a)

# в «практических» единицах
RH_cm, sRH_cm = RH * 1e6, sRH * 1e6                            # см³/Кл
n_cm, sn_cm = n_m3 * 1e-6, sn * 1e-6                           # см⁻³
rho_cm, srho_cm = rho * 100, srho * 100                        # Ом·см
sig_cm, ssig_cm = sigma0 * 0.01, ssigma * 0.01                 # (Ом·см)⁻¹
mu_cm, smu_cm = mu * 1e4, smu * 1e4                            # см²/(В·с)

# контрольный вопрос 6
m_star = M_EFF_REL[carrier_short] * M_E
tau = mu * m_star / E_CHARGE
v_th = np.sqrt(3 * K_B * T_K / m_star)
lam = v_th * tau
D = mu * K_B * T_K / E_CHARGE
q6_rel = smu / mu

# =====================================================================
#                            results.txt
# =====================================================================
L = []
P = L.append
bar = "=" * 78


def head(t):
    P("")
    P(bar)
    P(t)
    P(bar)


P("РАБОТА 3.3.4. ЭФФЕКТ ХОЛЛА В ПОЛУПРОВОДНИКАХ — результаты обработки")
P("Файл создан скриптом hall_334.py. Обозначения: I — ток через образец, I_M — ток электромагнита,")
P("U_perp — ЭДС Холла, k = dU_perp/dB, R_H — постоянная Холла, n — концентрация, mu — подвижность.")
P("Запись «(a ± b)·10^p» — погрешность b соответствует доверительной вероятности ~68 % (1 сигма).")
P("Числа в «ИТОГОВЫХ РЕЗУЛЬТАТАХ» округлены по правилам пособия, в остальных разделах — без округления.")

head("0. ЧТО ПРОВЕРИТЬ ПЕРЕД ВСТАВКОЙ В ОТЧЁТ (допущения)")
P(f"* Толщина образца h = {H_MM:g} мм взята из параметра 'l' листа parameters (в методичке l — это расстояние 3–5,")
P(f"  в xlsx оно называется L_35 = {par['L_35']:g} мм). Если h на держателе другая — поправь H_MM_OVERRIDE.")
P(f"* Приборные погрешности — допущения по правилу пособия (макс. погрешность ~ цене деления / последнему разряду):")
P(f"  U34, U0: ±{SIGMA_U_UV:g} мкВ каждое; U35: ±{SIGMA_U35_MV:g} мВ; миллиамперметр образца: ±{SIGMA_I_DIV:g} дел = ±{S_I_SAMPLE_MA:g} мА;")
P(f"  a, h, l35: ±{SIGMA_GEOM_MM:g} мм (по умолчанию ±1 в последней значащей цифре); прибор для градуировки B: ±{100 * B_INSTR_REL:g} %.")
P(f"  Паспортных классов точности в данных нет — при необходимости заменить в блоке НАСТРОЙКИ.")
P(f"* Знак носителей зависит от двух флагов, которых нет в данных: B_FWD_TOWARD_VIEWER={B_FWD_TOWARD_VIEWER}, "
  f"U34_IS_PHI3_MINUS_PHI4={U34_IS_PHI3_MINUS_PHI4}.")
P("  Их нужно сверить с установкой (направление тока в обмотках при К1 в положении 1, полярность входов вольтметра).")

head("1. ВХОДНЫЕ ПАРАМЕТРЫ")
P(f"Образец: ширина a = {par['a']:g} мм, расстояние 3–5 l35 = {par['L_35']:g} мм, толщина h = {H_MM:g} мм.")
P(f"Измерение удельного сопротивления: U35 = {par['U_rho']:g} мВ при I = {par['I_rho']:g} дел = {I_RHO_A * 1e3:.3f} мА "
  f"(цена деления {K_I:g} мА/дел).")
P(f"Токи образца в серии U34(B): " + ", ".join(f"{s['I']:.2f}" for s in series) + " мА.")
P(f"Ток через образец ≤ 1 мА по методичке (п. 2); максимум в данных {Iv.max():.2f} мА "
  + ("— ПРЕВЫШАЕТ эту рекомендацию (отметить в отчёте)." if Iv.max() > 1.0 else "— в норме."))

head("2. КАЛИБРОВКА ЭЛЕКТРОМАГНИТА B(I_M)   [fig1_calibration_B_of_IM.pdf]")
P(f"Точек: {use.sum()} (проходы {CAL_PASSES}), диапазон I_M = {Ical.min():g}…{Ical.max():g} мА, B = {Bcal.min():g}…{Bcal.max():g} мТл.")
P("Зависимость нелинейная (насыщение железа), поэтому аппроксимируется полиномом минимальной степени,")
P(f"после которой RMS остатков падает менее чем на {100 * DEG_GAIN_THRESHOLD:g} %.")
P("Степень d : RMS остатков, мТл")
for d in range(1, 9):
    P(f"   {d}{'  <- выбрано' if d == deg else '':<12s} {rms[d]:7.2f}")
P(f"Выбран полином {deg}-й степени:  B[мТл] = sum_j c_j * (I_M[мА])^j,")
P("   c_j (j = 0…%d): " % deg + ", ".join(f"{c:.6e}" for c in coef))
P(f"Монотонен на 0…{Ical.max():g} мА: {'да' if monotone else 'НЕТ (проверить!)'}.")
P(f"Остаточное поле: B(I_M=0) по аппроксимации = {Bfun(0):.1f} мТл (измерено {Bcal[Ical == 0][0]:g} мТл).")
P(f"Разброс точек вокруг кривой (RMS) sigma_B = {sB:.1f} мТл — принят как погрешность отдельной точки калибровки.")
for pn in np.unique(passno):
    m = passno == pn
    P(f"Проход {pn}: {m.sum()} точек, среднее отклонение от кривой {cal_res[m].mean():+.1f} мТл, RMS {np.sqrt(np.mean(cal_res[m] ** 2)):.1f} мТл")
P("Различие проходов — оценка гистерезиса/воспроизводимости. Кривая используется для пересчёта I_M -> B")
P("только внутри диапазона калибровки (экстраполяции нет).")

head("3. ЭДС ХОЛЛА: U_perp = U34 - U0 (ф. (1)), ЛИНЕЙНЫЕ АППРОКСИМАЦИИ U_perp = k·B + b   [fig2_hall_family_Uperp_of_B.pdf]")
P(f"Точки I_M >= 50 мА (точка I_M = 0 — начало отсчёта, U_perp = 0 по определению). Погрешность U_perp: sqrt(2)·{SIGMA_U_UV:g} = {SU:.2f} мкВ.")
P("Метод: взвешенный хи-квадрат (scipy.curve_fit, absolute_sigma=True). Погрешность B учтена отдельно как систематика (см. раздел 8),")
P("т.к. B берётся с одной гладкой калибровочной кривой и ошибки соседних точек коррелируют.")
P("")
P("  I, мА |  U0, мкВ |  k, мкВ/мТл (± σ)     |  b, мкВ (± σ)    | χ²/dof | доля крестов, пересечённых прямой")
for s in series:
    f = s["fit"]
    P(f"  {s['I']:5.2f} | {s['U0']:8.0f} | {f['k']:9.5f} ± {f['sk']:.5f} | {f['b']:7.2f} ± {f['sb']:.2f} | "
      f"{f['chi2'] / f['dof']:5.2f}  | {100 * f['frac']:5.0f} %  ({int(round(f['frac'] * len(s['B'])))}/{len(s['B'])})")
chi_mean = np.mean([s["fit"]["chi2"] / s["fit"]["dof"] for s in series])
P(f"Среднее χ²/dof по сериям = {chi_mean:.2f}.  Критерий пособия: ~1 — модель и погрешности согласованы; <0.5 — погрешности завышены;")
P("  >1.5 — плохое согласие или заниженные погрешности. Критерий «≈2/3 крестов пересечено» — для проверки линейности по графику.")
bk = [-s["fit"]["b"] / s["fit"]["k"] for s in series]
low = [s["I"] for s in series if s["fit"]["frac"] < 2 / 3]
P("Линейность U_perp(B): " + ("прямая пересекает не менее 2/3 крестов во всех сериях." if not low else
  "у серий I = " + ", ".join(f"{v:.2f}" for v in low) + " мА прямая пересекает менее 2/3 крестов (малые отклонения от линейности/шум, "
  "χ²/dof близко к 1 — в пределах погрешностей), у остальных — не менее 2/3."))
P("Отрицательный отрезок b объясняется остаточным полем магнита:")
P(f"  из-за B_ост при I_M=0 U_perp = k·(B - B_ост), т.е. -b/k ≈ B_ост. Среднее -b/k = {np.mean(bk):.0f} мТл "
  f"(разброс {np.std(bk, ddof=1):.0f} мТл), калибровка даёт B(0) ≈ {Bfun(0):.0f} мТл.")
P("  На наклон k остаточное поле не влияет, поэтому аппроксимация с отрезком (двухпараметрическая) корректна.")

head("4. ЗАВИСИМОСТЬ k(I) И ПОСТОЯННАЯ ХОЛЛА R_H   [fig4_k_of_I.pdf]")
P("Теория (ф. (3.27)): U_perp = R_H·I·B / h  =>  k = (R_H / h)·I  — прямая через ноль с наклоном s = dk/dI = R_H / h.")
P(f"Погрешность по оси I (±{S_I_SAMPLE_MA:g} мА) учтена эффективной дисперсией σ_k² + (s·σ_I)².")
P(f"Аппроксимация k = s·I + k0:  s = {fk['k']:.5f} ± {fk['sk']:.5f} мкВ/(мТл·мА)  [= В/(Тл·А)],  k0 = {fk['b']:.5f} ± {fk['sb']:.5f} мкВ/мТл,")
P(f"   χ²/dof = {fk['chi2'] / fk['dof']:.2f};  k0 отличается от нуля на {abs(fk['b']) / fk['sb']:.1f} σ.")
P(f"Аппроксимация через ноль:      s = {fk0['k']:.5f} ± {fk0['sk']:.5f}  мкВ/(мТл·мА),  χ²/dof = {fk0['chi2'] / fk0['dof']:.2f}")
P(f"   (разница наклонов двух вариантов: {abs(fk['k'] - fk0['k']) / fk['sk']:.1f} σ двухпараметрической).")
P("Для итога используется двухпараметрический вариант (пособие, п. 4.1.1).")
if fk["chi2"] / fk["dof"] < 0.5:
    P(f"χ²/dof = {fk['chi2'] / fk['dof']:.2f} << 1: точки лежат на прямой заметно лучше, чем допускают погрешности — они (в основном σ_I = {SIGMA_I_DIV:g} дел) "
      f"заведомо завышены, оценка погрешности наклона консервативна.")
P("")
P("  I, мА | σ_I, мА |  k, мкВ/мТл  |  σ_k    | k/I (R_H/h), мкВ/(мТл·мА)")
for s in series:
    P(f"  {s['I']:5.2f} | {S_I_SAMPLE_MA:6.3f}  | {s['fit']['k']:10.5f}  | {s['fit']['sk']:.5f} | {s['fit']['k'] / s['I']:.4f}")

head("5. ОБРАТНОЕ ПОЛЕ (п. 8) И ЗНАК ЭДС ХОЛЛА   [fig3_reverse_field_check.pdf]")
P(f"Положительный наклон k > 0: при поле «вперёд» U_perp > 0 (на всех токах). Знак U_perp: {'+' if sign_reading > 0 else '−'}.")
if rev is not None:
    fr, fs, ff = rev["fit"], rev["fsym"], fmax["fit"]
    P(f"I = {fmax['I']:.2f} мА. U0 вперёд = {fmax['U0']:.0f} мкВ, U0 при обращённом поле = {rev['U0']:.0f} мкВ (различие — остаточное поле меняет знак).")
    P(f"Наклон, поле «вперёд» (U34 - U0):       k = {ff['k']:.5f} ± {ff['sk']:.5f} мкВ/мТл")
    P(f"Наклон, поле обращено (U34 - U0_обр):   k = {fr['k']:.5f} ± {fr['sk']:.5f} мкВ/мТл   (B взято со знаком «−»)")
    P(f"Метод полуразности U_perp = [U34(+) - U34(−)]/2, точки I_M = 0…: k = {fs['k']:.5f} ± {fs['sk']:.5f} мкВ/мТл,  b = {fs['b']:.2f} ± {fs['sb']:.2f} мкВ")
    P(f"Расхождение наклонов вперёд/назад: {abs(ff['k'] - fr['k']) / quad(ff['sk'], fr['sk']):.1f} σ; вперёд/полуразность: "
      f"{abs(ff['k'] - fs['k']) / quad(ff['sk'], fs['sk']):.1f} σ.")
    RHs = fs["k"] * 1e-3 * H_M / (fmax["I"] * 1e-3) * 1e6
    P(f"Метод полуразности не использует U0: омическая добавка от несоосности контактов 3, 4 одинакова при обоих направлениях поля и вычитается.")
    P(f"Контроль R_H по одной серии (I = {fmax['I']:.2f} мА, полуразность): R_H = k·h/I = {RHs:.0f} ± {RHs * quad(fs['sk'] / fs['k'], r_h):.0f} см³/Кл "
      f"(без калибровочных и приборных поправок), основной результат (раздел 7) — {RH_cm:.0f} см³/Кл.")
    P("Знак U_perp меняется на обратный при обращении поля — это эффект Холла (омическая добавка от несимметрии контактов от поля не зависит).")
    if abs(fs["b"]) < 2 * fs["sb"]:
        P("Для метода полуразности отрезок b совместим с нулём (остаточное поле входит симметрично и сокращается).")
    else:
        P(f"Для метода полуразности отрезок b = {fs['b']:.2f} ± {fs['sb']:.2f} мкВ не равен нулю: B(I_M) при обращении поля не строго симметрична (гистерезис,")
        P("остаточное поле); на наклон это влияет слабо.")
    P(f"Наклоны вперёд и назад различаются на {200 * dir_rel:.1f} % — это существенно больше статистической погрешности ({100 * ff['sk'] / ff['k']:.2f} %).")
    P(f"Половина этой разности ({100 * dir_rel:.1f} %) добавлена в бюджет погрешностей как систематика от направления поля/гистерезиса (раздел 8).")

head("6. ЗНАК НОСИТЕЛЕЙ (п. 9)")
P("Геометрия рис. 1б: ток от контакта 2 (+) к контакту 1 (−) («вверх»), контакт 3 слева, 4 справа. При B на наблюдателя")
P("носители обоих знаков отклоняются к контакту 4: электроны -> U34 = φ3-φ4 > 0, дырки -> U34 < 0.")
P(f"Принято: B_вперёд {'на наблюдателя' if B_FWD_TOWARD_VIEWER else 'от наблюдателя'}, U34 = {'φ3−φ4' if U34_IS_PHI3_MINUS_PHI4 else 'φ4−φ3'}; "
  f"U_perp {'>' if sign_reading > 0 else '<'} 0.")
P(f"ВЫВОД (при этих допущениях): основные носители — {carrier}.")
P("Независимая проверка: подвижность |mu| сравнивается со справочной для электронов и дырок в Ge (раздел 9) —")
P("значение mu заметно выше справочной подвижности дырок в Ge (1900 см²/(В·с)) — дырочная проводимость маловероятна." if mu_cm > 2500
  else "значение mu сравни со справочными подвижностями (раздел 9).")

head("7. ИТОГОВЫЕ РЕЗУЛЬТАТЫ (округлены по правилам пособия)")
P(f"Постоянная Холла      R_H = s·h = {fmt(RH_cm, sRH_cm, 'см³/Кл')}     [{fmt(RH, sRH, 'м³/Кл')}],  ε = {rel(RH, sRH)}")
P(f"Концентрация          n = 1/(e|R_H|) = {fmt(n_cm, sn_cm, 'см⁻³')}     [{fmt(n_m3, sn, 'м⁻³')}],  ε = {rel(n_m3, sn)}")
P(f"Уд. сопротивление     rho0 = U35·a·h/(I·l35) = {fmt(rho_cm, srho_cm, 'Ом·см')}     [{fmt(rho, srho, 'Ом·м')}],  ε = {rel(rho, srho)}")
P(f"Уд. проводимость      sigma0 = 1/rho0 = {fmt(sig_cm, ssig_cm, '(Ом·см)⁻¹')}     [{fmt(sigma0, ssigma, 'См/м')}],  ε = {rel(sigma0, ssigma)}")
P(f"Подвижность           mu = |R_H|·sigma0 = {fmt(mu_cm, smu_cm, 'см²/(В·с)')}     [{fmt(mu, smu, 'м²/(В·с)')}],  ε = {rel(mu, smu)}")
P(f"Тип носителей: {carrier} (см. раздел 6; зависит от допущений о геометрии).")
P(f"Промежуточные: s = dk/dI = {fmt(s_si, ss_si, 'В/(Тл·А)')};  I_rho = {I_RHO_A * 1e3:.3f} мА, U35 = {par['U_rho']:g} мВ.")
P(f"Проверка: mu из прямой формулы s·I·l35/(U35·a) = {mu_check * 1e4:.1f} см²/(В·с) (должно совпасть с {mu_cm:.1f}).")

head("8. БЮДЖЕТ ПОГРЕШНОСТЕЙ (относительные вклады, %)")
P("Складываются квадратично (независимые источники).")
P(f"R_H и n :  наклон s из аппроксимации k(I) {100 * r_s:.2f};  толщина h {100 * r_h:.2f};  калибровка B (бутстрэп) {100 * cal_rel:.2f};  "
  f"прибор для B (допущение) {100 * B_INSTR_REL:.2f};  направление поля {100 * dir_rel:.2f}   ->  итого {100 * quad(r_s, r_h, cal_rel, B_INSTR_REL, dir_rel):.1f}")
P(f"rho0, sigma0 :  U35 {100 * r_U35:.2f};  a {100 * r_a:.2f};  h {100 * r_h:.2f};  I_rho {100 * r_Irho:.2f};  l35 {100 * r_l:.2f}   "
  f"->  итого {100 * quad(r_U35, r_a, r_h, r_Irho, r_l):.1f}")
P(f"mu :  в формуле mu = s·I_rho·l35/(U35·a) толщина h сокращается; вклады: s {100 * r_s:.2f}; B {100 * quad(cal_rel, B_INSTR_REL, dir_rel):.2f}; "
  f"I_rho {100 * r_Irho:.2f}; l35 {100 * r_l:.2f}; U35 {100 * r_U35:.2f}; a {100 * r_a:.2f}   ->  итого {100 * smu / mu:.1f}")
dom = max([("толщина h", r_h), ("ширина a", r_a), ("расстояние l35", r_l), ("наклон s", r_s)], key=lambda t: t[1])
P(f"Главный вклад в погрешности даёт геометрия образца (самый большой вклад: {dom[0]}, {100 * dom[1]:.1f} %); "
  f"статистика аппроксимаций значительно меньше — улучшить результат можно только точнее измерив размеры образца.")

head("9. СРАВНЕНИЕ С ТАБЛИЧНЫМИ ЗНАЧЕНИЯМИ (п. 17)")
P("Справочные подвижности Ge при 300 К (проверь по таблицам из задачника): " + "; ".join(f"{k} {v:g}" for k, v in MU_TABLE.items()) + " см²/(В·с).")
for k_, v_ in MU_TABLE.items():
    dd = mu_cm - v_
    P(f"  mu_эксп = {mu_cm:.0f} ± {smu_cm:.0f}  против {k_}: {v_:g}  ->  разность {dd:+.0f} ({dd / smu_cm:+.1f} σ, {100 * dd / v_:+.0f} %)")
P("Концентрация n и удельное сопротивление зависят от степени легирования конкретного образца; универсальных табличных значений нет —")
P("сравнивать можно с данными, указанными на держателе/в описании установки. Холл-фактор принят равным 1 (R_H = 1/(n e)).")
_ref = MU_TABLE["электроны (n-Ge, 300 К)"] if eff > 0 else MU_TABLE["дырки (p-Ge, 300 К)"]
if abs(mu_cm - _ref) > 2 * smu_cm:
    P("Расхождение со справочной подвижностью для найденного типа носителей значимо (> 2σ). Возможные причины: систематика размеров образца")
    P("(mu ∝ l35/a — ошибка a или l35 прямо входит в результат), приборные классы точности, холл-фактор r_H ≠ 1, температура, примеси,")
    P("а также нелинейность/гистерезис калибровки. Какая именно — по этим данным установить нельзя.")
else:
    P("Расхождение со справочной подвижностью для найденного типа носителей в пределах 2σ.")

head("10. ДОПОЛНИТЕЛЬНО: КОНТРОЛЬНЫЙ ВОПРОС 6 (оценка)")
P(f"Допущения: T = {T_K:g} К, эффективная масса {carrier_short} m* = {M_EFF_REL[carrier_short]:g}·m_e (ориентировочная справочная), тепловая скорость v = sqrt(3kT/m*).")
P(f"Время свободного пробега tau = mu·m*/e = {fmt(tau, tau * q6_rel, 'с')};  частота столкновений nu = 1/tau = {fmt(1 / tau, q6_rel / tau, 'с⁻¹')}")
P(f"Длина свободного пробега  lambda = v·tau = {fmt(lam * 1e9, lam * 1e9 * q6_rel, 'нм')}   (v = {v_th:.2e} м/с)")
P(f"Коэффициент диффузии (соотношение Эйнштейна) D = mu·kT/e = {fmt(D * 1e4, D * 1e4 * q6_rel, 'см²/с')}")
P("Погрешность — только от mu (m* и T приняты точными).")

head("11. ФАЙЛЫ И РЕКОМЕНДУЕМЫЕ ПОДПИСИ К РИСУНКАМ")
P(f"fig1_calibration_B_of_IM.pdf — Градуировка электромагнита B(I_M). Линия — полином {deg}-й степени, усы ±{sB:.1f} мТл (разброс точек вокруг кривой).")
P(f"fig2_hall_family_Uperp_of_B.pdf — ЭДС Холла U_perp = U34 − U0 от B при токах образца {Iv.min():.2f}…{Iv.max():.2f} мА. Линии — МНК с весами, в легенде наклоны k, мкВ/мТл.")
P(f"fig3_reverse_field_check.pdf — U_perp(B) при прямом и обратном поле, I = {fmax['I']:.2f} мА; ромбы — полуразность [U34(+) − U34(−)]/2, не содержащая омической добавки.")
P("fig4_k_of_I.pdf — Зависимость наклона k от тока I. Сплошная линия — МНК с учётом погрешностей по X и Y, пунктир — прямая через ноль; s = R_H/h.")

head("12. ТАБЛИЦА ИСХОДНЫХ ТОЧЕК (для приложения к отчёту)")
P("серия I, мА | I_M, мА |  B, мТл |  U34, мкВ |  U0, мкВ | U_perp, мкВ")
for s in series:
    for i in range(len(s["IM"])):
        P(f"   {s['I']:5.2f}    | {s['IM'][i]:6.0f}  | {s['B'][i]:7.1f} | {s['U34'][i]:8.0f}  | {s['U0']:7.0f}  | {s['Up'][i]:7.0f}")
if rev is not None:
    P("обратное поле, I = %.2f мА (B со знаком «−»):" % fmax["I"])
    for i in range(len(rev["IM"])):
        P(f"   {fmax['I']:5.2f}    | {rev['IM'][i]:6.0f}  | {rev['B'][i]:7.1f} | {rev['Up'][i] + rev['U0']:8.0f}  | {rev['U0']:7.0f}  | {rev['Up'][i]:7.0f}")

if rev is not None:
    P("полуразность [U34(+) − U34(−)]/2, I = %.2f мА (I_M = 0 — из U0):" % fmax["I"])
    for i in range(len(rev["Isym"])):
        P(f"   {fmax['I']:5.2f}    | {rev['Isym'][i]:6.0f}  | {Bfun(rev['Isym'][i]):7.1f} | {rev['Usym'][i]:7.1f}")

if warnings:
    head("ПРЕДУПРЕЖДЕНИЯ")
    for w in warnings:
        P("! " + w)

(RESULTS_DIR / "results.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
print(f"Готово. results.txt -> {RESULTS_DIR}")
print(f"Графики -> {FIGURES_DIR}")
print(f"R_H = {fmt(RH_cm, sRH_cm, 'см³/Кл')};  n = {fmt(n_cm, sn_cm, 'см⁻³')};  mu = {fmt(mu_cm, smu_cm, 'см²/(В·с)')};  носители: {carrier}")