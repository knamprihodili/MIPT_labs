#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Лаб. 3.5.2: обработка зондовых ВАХ двойного зонда в плазме ВЧ-разряда.
Нумерация формул и пунктов - по «Лабораторному практикуму по общей физике», т. 2 (МФТИ),
раздел V «Плазма. Газовый разряд» и описание работы 3.5.2.

Вход:   data/vah_<P>.csv - оцифрованные ВАХ в делениях экрана (scripts/digitize_vah.py)
Выход:  figures/vah_<P>.pdf        - ВАХ I(U) для каждого давления: асимптоты и касательная в нуле
        figures/vah_all.pdf        - все ВАХ на одном графике (центрированы)
        figures/plasma_vs_p.pdf    - T_e, n_e, omega_p, r_D(e) в зависимости от давления (сетка 2x2)
        data/results.csv           - сводная таблица
        data/results_table.tex     - та же таблица для LaTeX

П. 12 (масштабы):
  X (CH1): k_X В/дел на входе осциллографа, перед ним делитель Д 1:100  ->  K_U = 100 k_X
  Y (CH2): k_Y В/дел, сигнал с резистора R = 100 кОм                      ->  K_I = k_Y / R
  Положения переключателей VOLTS/DIV (записаны в ходе работы, ручки VAR - в положении CAL):
      P, торр   CH1 (X)    CH2 (Y)
      8e-2      0.1 В/дел  2 В/дел
      9e-2      0.1 В/дел  2 В/дел
      1e-1      0.1 В/дел  1 В/дел
      1.5e-1    0.1 В/дел  1 В/дел
      2e-1      0.1 В/дел  0.5 В/дел

П. 13-14 (T_e и I_iн) - как в п. 3.3 Введения:
  I_iн - пересечение асимптот (с наклоном, см. рис. 5.8) с осью U = 0; асимптоты - прямые
  по крайним четвертям размаха U. Наклон в нуле - прямая по центральным точкам. Затем
  kT_e = (1/2) e I_iн / (dI/dU)|_{U=0}                                    (5.26)
  Концентрация - по формуле Бома (5.22): I_iн = 0.4 n e S sqrt(2kT_e/m_i), S = pi d l,
  ионы однозарядные, m_i = 28 а.е.м. (N2+, трубка заполнена воздухом).
П. 15-17: omega_p (5.1), r_De и r_Di (5.9), r_D (5.10) при T_i = 300 К, N_D (5.13),
  alpha = n / n_газа, n_газа = P / kT.

Погрешности - методом Монте-Карло: масштабы X и Y - 3 % (паспорт GOS-620), делитель и
резистор - 1 %, давление - 10 % (градуировочная кривая вакуумметра), а также разброс
результата при разном выборе участков для асимптот и касательной.

Нужно: numpy, matplotlib  (pip install numpy matplotlib)
Запуск:  python scripts/analyze_plasma.py   (из корня папки лабы)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# крупные шрифты: подписи осей 20, числа на осях 15, заголовки 18, легенда 12
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 20,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 12,
})

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
FIG = os.path.join(ROOT, "figures")

# ---------------- константы установки и опыта ----------------
P_TORR = {"8e-2": 8e-2, "9e-2": 9e-2, "1e-1": 1e-1, "1.5e-1": 1.5e-1, "2e-1": 2e-1}
# (CH1 X, CH2 Y), В/дел на входах осциллографа
OSC = {"8e-2": (0.1, 2.0), "9e-2": (0.1, 2.0), "1e-1": (0.1, 1.0), "1.5e-1": (0.1, 1.0), "2e-1": (0.1, 0.5)}
DIV = 100.0            # делитель Д 1:100 в канале X
R_SENS = 100e3         # Ом, датчик тока
KX = {p: kx * DIV for p, (kx, ky) in OSC.items()}        # В/дел на зонде
KY = {p: ky / R_SENS for p, (kx, ky) in OSC.items()}     # А/дел

d_probe, l_probe = 0.5e-3, 10e-3          # м
S = np.pi * d_probe * l_probe             # м^2, площадь одного зонда
A_TUBE = 0.75                             # см, радиус трубки
AMU = 1.66e-27
M_I_AMU = 28                              # N2+
M_I = M_I_AMU * AMU
E = 1.602e-19
K_B = 1.381e-23
T_I = 300.0                               # К
EV_K = E / K_B                            # 1 эВ в кельвинах
F_HF = 10e6                               # Гц, частота ВЧ-генератора
E_CGS, KB_CGS = 4.803e-10, 1.381e-16

SIG_SCALE = 0.03       # отн. погрешность коэффициентов отклонения осциллографа
SIG_DIVR = 0.01        # делитель и резистор
SIG_P = 0.10           # давление
Y_CLIP = 3.9           # дел.: выше луч упирается в край экрана (P = 1e-1)
N_MC = 20000
rng = np.random.default_rng(352)


def load(p):
    d = np.loadtxt(os.path.join(DATA, f"vah_{p}.csv"), delimiter=",", skiprows=1)
    # центр по X: синусоида ЗГ симметрична относительно U = 0, а при одинаковых зондах
    # центр антисимметрии ВАХ - U = 0. Берём середину размаха луча по X (с учётом
    # срезанного краем экрана участка); смещение по Y (I0) - середина между асимптотами.
    kx, ky = KX[p], KY[p]
    U0 = 0.5 * (d[:, 0].min() + d[:, 0].max()) * kx
    m = d[:, 1] < Y_CLIP
    U = d[m, 0] * kx
    I = d[m, 1] * ky * 1e6                    # мкА
    sI = np.maximum((d[m, 3] - d[m, 2]) / 2, 0.05) * ky * 1e6   # полуширина луча/петли
    return U, I, sI, U0


def asymptote_method(U, I, U0, f_sat=0.75, f_c=0.2):
    """Способ из п. 3.3 Введения: асимптоты (крайние четверти размаха) и касательная в нуле."""
    x = U - U0
    span = min(x.max(), -x.min())
    hi = x > f_sat * x.max()
    lo = x < f_sat * x.min()
    ah = np.polyfit(x[hi], I[hi], 1)
    al = np.polyfit(x[lo], I[lo], 1)
    Iin = 0.5 * (np.polyval(ah, 0) - np.polyval(al, 0))
    I0 = 0.5 * (np.polyval(ah, 0) + np.polyval(al, 0))   # центр характеристики по току
    c = abs(x) < f_c * span
    k = np.polyfit(x[c], I[c], 1)
    T = 0.5 * Iin / k[0]
    return dict(Iin=Iin, I0=I0, slope=k[0], T=T, ah=ah, al=al, k=k, dU=Iin / k[0])


def plasma_params(T_eV, Iin_A, P_torr):
    """П. 14-17. T_eV, Iin_A, P_torr - массивы (Монте-Карло)."""
    v = np.sqrt(2 * T_eV * E / M_I)
    n_cm3 = Iin_A / (0.4 * E * S * v) * 1e-6                           # (5.22)
    wp = np.sqrt(4 * np.pi * n_cm3 * E_CGS ** 2 / 9.109e-28)            # (5.1), рад/с
    rDe = np.sqrt(KB_CGS * T_eV * EV_K / (4 * np.pi * n_cm3 * E_CGS ** 2))   # (5.9), см
    rDi = np.sqrt(KB_CGS * T_I / (4 * np.pi * n_cm3 * E_CGS ** 2))
    rD = (rDe ** -2 + rDi ** -2) ** -0.5                                 # (5.10)
    ND = n_cm3 * 4 / 3 * np.pi * rD ** 3                                 # (5.13)
    n_gas = P_torr * 133.322 / (K_B * T_I) * 1e-6                       # см^-3
    alpha = n_cm3 / n_gas
    return dict(n=n_cm3, wp=wp, rDe=rDe, rD=rD, ND=ND, alpha=alpha, n_gas=n_gas)


def summarize(a):
    q = np.percentile(a, [16, 50, 84])
    return q[1], 0.5 * (q[2] - q[0])


KEYS = ("n", "wp", "rDe", "rD", "ND", "alpha", "n_gas")


def P_label(p):
    m, e = p.split("e")
    return fr"{m}\cdot 10^{{{int(e)}}}"


def main():
    os.makedirs(FIG, exist_ok=True)
    rows = []
    for p, P in P_TORR.items():
        U, I, sI, U0 = load(p)
        am = asymptote_method(U, I, U0)
        I0 = am["I0"]
        # произвол построения: где проводить асимптоты и касательную
        var = [asymptote_method(U, I, U0, fs, fc) for fs in (0.65, 0.75, 0.85) for fc in (0.1, 0.2, 0.3)]
        sT_sys = np.std([v["T"] for v in var]) / am["T"]
        sI_sys = np.std([v["Iin"] for v in var]) / am["Iin"]

        # --- Монте-Карло ---
        sx = 1 + rng.normal(0, np.hypot(SIG_SCALE, SIG_DIVR), N_MC)
        sy = 1 + rng.normal(0, np.hypot(SIG_SCALE, SIG_DIVR), N_MC)
        sp = 1 + rng.normal(0, SIG_P, N_MC)
        e_sys = rng.normal(0, 1, N_MC)          # T и I_iн при изменении построения коррелированы
        Ta = am["T"] * sx * np.exp(sT_sys * e_sys)
        Ia = am["Iin"] * 1e-6 * sy * np.exp(sI_sys * e_sys)
        pa = plasma_params(Ta, Ia, P * sp)

        r = dict(p=p, P=P, U0=U0, I0=I0, am=am, Iin=summarize(Ia * 1e6), Te=summarize(Ta),
                 span=(U.max() - U0) / am["dU"])
        for key in KEYS:
            r[key] = summarize(pa[key])
        rows.append(r)

        # --- график ВАХ: отдельный файл на каждое давление ---
        fig, a = plt.subplots(figsize=(8, 6.2))
        a.fill_between(U, I - sI, I + sI, color="0.75", lw=0, label="ширина луча / петли")
        a.plot(U, I, "o", ms=3, color="tab:blue", label="оцифровка (средняя линия)")
        xa = np.linspace(0, U.max() - U0 + 2, 50)
        a.plot(U0 + xa, I0 + np.polyval(am["ah"], xa) - am["ah"][1] + am["Iin"], "--", color="tab:green", lw=1.5)
        xb = np.linspace(U.min() - U0 - 2, 0, 50)
        a.plot(U0 + xb, I0 + np.polyval(am["al"], xb) - am["al"][1] - am["Iin"], "--", color="tab:green", lw=1.5,
               label="асимптоты")
        xc = np.linspace(-1.3 * am["dU"], 1.3 * am["dU"], 10)
        a.plot(U0 + xc, I0 + am["k"][0] * xc, ":", color="k", lw=1.8, label="касательная в нуле")
        a.plot([U0, U0 + am["dU"]], [I0 + am["Iin"]] * 2, "-", color="tab:purple", lw=2.2)
        a.plot(U0, I0 + am["Iin"], "s", color="tab:purple", ms=6)
        a.plot(U0 + am["dU"], I0 + am["Iin"], "D", color="tab:purple", ms=6,
               label=fr"$\Delta U = 2kT_e/e$: $kT_e={am['T']:.1f}$ эВ")
        a.axhline(I0, color="0.4", lw=0.7); a.axvline(U0, color="0.4", lw=0.7)
        a.set_title(fr"$P = {P_label(p)}$ торр")
        a.set_xlabel(r"$U$, В"); a.set_ylabel(r"$I$, мкА")
        a.grid(lw=0.4, alpha=0.6)
        a.set_xlim(-55, 55); a.set_ylim(-5.5 * KY[p] * 1e6, 5.5 * KY[p] * 1e6)
        a.legend(loc="upper left", fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, f"vah_{p}.pdf"))
        plt.close(fig)

    # --- все ВАХ на одном графике ---
    fig, a = plt.subplots(figsize=(8, 6.2))
    for p, P in P_TORR.items():
        U, I, sI, U0 = load(p)
        r = [rr for rr in rows if rr["p"] == p][0]
        a.plot(U - U0, I - r["I0"], "-", lw=1.8, label=fr"${P_label(p)}$ торр")
    a.set_title("Все ВАХ (центрированы)")
    a.set_xlabel(r"$U-U_0$, В"); a.set_ylabel(r"$I-I_0$, мкА")
    a.grid(lw=0.4, alpha=0.6); a.legend(loc="upper left")
    a.set_xlim(-55, 55)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "vah_all.pdf"))
    plt.close(fig)

    plot_vs_p(rows)
    write_tables(rows)
    print_report(rows)


def plot_vs_p(rows):
    P = np.array([r["P"] for r in rows])
    fig, axs = plt.subplots(2, 2, figsize=(13, 10))
    axs = axs.ravel()

    def ep(a, key, scale=1, **kw):
        v = np.array([r[key][0] for r in rows]) * scale
        s = np.array([r[key][1] for r in rows]) * scale
        a.errorbar(P, v, yerr=s, fmt="o", ms=7, capsize=4, **kw)
    ep(axs[0], "Te")
    axs[0].set_ylabel(r"$kT_e$, эВ"); axs[0].set_title("Температура электронов")
    ep(axs[1], "n", 1e-9)
    axs[1].set_ylabel(r"$n_e$, $10^{9}$ см$^{-3}$"); axs[1].set_title("Концентрация электронов")
    ep(axs[2], "wp", 1e-6, color="tab:red")
    axs[2].axhline(2 * np.pi * F_HF * 1e-6, color="k", ls="--", lw=1.2)
    axs[2].text(0.065, 2 * np.pi * F_HF * 1e-6 + 60, r"$\omega_{\rm ВЧ} = 2\pi\cdot 10$ МГц", fontsize=13)
    axs[2].set_ylabel(r"$\omega_p$, $10^6$ рад/с"); axs[2].set_title("Плазменная частота")
    ep(axs[3], "rDe", 1e4, color="tab:purple", label=r"$r_{De}$")
    ep(axs[3], "rD", 1e4, color="tab:green", label=r"$r_D \approx r_{Di}$")
    axs[3].set_yscale("log"); axs[3].set_ylim(10, 2000)
    axs[3].set_ylabel("мкм"); axs[3].set_title("Дебаевские длины")
    for a in axs:
        a.set_xlabel(r"$P$, торр"); a.grid(lw=0.4, alpha=0.6); a.set_xlim(0.06, 0.22)
    axs[3].legend(fontsize=13)
    axs[0].set_ylim(0, 13); axs[1].set_ylim(bottom=0); axs[2].set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "plasma_vs_p.pdf"))
    plt.close(fig)


def fmt(v, s):
    """Округление по погрешности (1-2 значащие цифры)."""
    e = int(np.floor(np.log10(s)))
    d = max(-e + (1 if s / 10 ** e < 3 else 0), 0)
    return f"{round(v, d):.{d}f}", f"{round(s, d):.{d}f}"


def sci(v, s):
    e = int(np.floor(np.log10(abs(v))))
    a, b = fmt(v / 10 ** e, s / 10 ** e)
    return a, b, e


def write_tables(rows):
    cols = ["Iin", "Te"] + list(KEYS)
    with open(os.path.join(DATA, "results.csv"), "w") as f:
        f.write("P_torr," + ",".join(f"{c},s_{c}" for c in cols) + "\n")
        for r in rows:
            vals = [r["P"]] + [x for c in cols for x in r[c]]
            f.write(",".join(f"{x:.5g}" for x in vals) + "\n")
    # единицы в CSV: I в мкА, T в эВ, n в см^-3, omega_p в рад/с, r_D в см

    L = []
    for r in rows:
        Ii, sIi = fmt(*r["Iin"]); Te, sTe = fmt(*r["Te"])
        n, sn, en = sci(*r["n"]); w, sw, ew = sci(*r["wp"])
        rde, srde = fmt(r["rDe"][0] * 1e4, r["rDe"][1] * 1e4)
        rd, srd = fmt(r["rD"][0] * 1e4, r["rD"][1] * 1e4)
        ND, sND = fmt(*r["ND"]); al, sal, ea = sci(*r["alpha"])
        L.append(fr"${P_label(r['p'])}$ & ${Ii}\pm{sIi}$ & ${Te}\pm{sTe}$ & "
                 fr"$({n}\pm{sn})\cdot10^{{{en}}}$ & $({w}\pm{sw})\cdot10^{{{ew}}}$ & ${rde}\pm{srde}$ & "
                 fr"${rd}\pm{srd}$ & ${ND}\pm{sND}$ & $({al}\pm{sal})\cdot10^{{{ea}}}$ \\")
    with open(os.path.join(DATA, "results_table.tex"), "w") as f:
        f.write("\n".join(L) + "\n")


def print_report(rows):
    print(f"S = {S*1e6:.2f} мм^2, m_i = {M_I_AMU} а.е.м., 1 эВ = {EV_K:.0f} К")
    for p in P_TORR:
        print(f"  {p}: K_U = {KX[p]:.0f} В/дел, K_I = {KY[p]*1e6:.0f} мкА/дел")
    for r in rows:
        am = r["am"]
        print(f"\nP = {r['P']} торр  (U0 = {r['U0']:.1f} В, I0 = {r['I0']:.1f} мкА, U_max - U0 = {r['span']:.1f} dU)")
        print(f"  I_iн = {r['Iin'][0]:.2f} ± {r['Iin'][1]:.2f} мкА, dI/dU = {am['slope']:.3f} мкА/В, "
              f"dU = {am['dU']:.1f} В, kTe = {r['Te'][0]:.2f} ± {r['Te'][1]:.2f} эВ (T = {r['Te'][0]*EV_K:.3g} К)")
        print(f"  n = {r['n'][0]:.3g} ± {r['n'][1]:.2g} см^-3, wp = {r['wp'][0]:.3g} ± {r['wp'][1]:.2g} рад/с, "
              f"r_De = {r['rDe'][0]*1e4:.0f} ± {r['rDe'][1]*1e4:.0f} мкм, r_D = {r['rD'][0]*1e4:.1f} ± {r['rD'][1]*1e4:.1f} мкм, "
              f"N_D = {r['ND'][0]:.0f} ± {r['ND'][1]:.0f}, alpha = {r['alpha'][0]:.3g} ± {r['alpha'][1]:.2g} "
              f"(n_газа = {r['n_gas'][0]:.3g} см^-3)")


if __name__ == "__main__":
    main()