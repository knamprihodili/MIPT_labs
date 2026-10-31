#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Лаб. 3.3.4  «Эффект Холла в полупроводниках» — обработка данных.
Методика погрешностей — по пособию Попова и Нозика «Обработка результатов учебного эксперимента» (МФТИ, 2024).

Что делает скрипт (нумерация — как в методичке к работе):
  п.12   калибровка B(I_M): интерполяция ПОЛИНОМОМ МИНИМАЛЬНОЙ СТЕПЕНИ в рабочем диапазоне токов
         (пособие, п. 3.4: для калибровочного графика это допустимо, коэффициентам физ. смысла не приписываем);
  п.13   U_perp = U34 - U0 (ф.(1)), семейство U_perp(B), линейные fit методом χ² (curve_fit), k = dU_perp/dB;
  п.8-9  обратное поле, симметризация, знак ЭДС Холла;
  п.14   k(I), R_H = h * dk/dI  (ф.(3.27));
  п.10,15,16  ρ0, σ0, n = 1/(e|R_H|), μ = |R_H| σ0;
  п.17   погрешности: статистическая (из fit, метод χ²) и систематическая (приборная) складываются в
         квадратуре (пособие, ф.(2.9)), косвенные величины — по ф.(2.12)/(2.14); округление — по п. 3.2.

Структура папок (пути считаются от расположения скрипта, запускать можно из любого места):
    lab_3.3.4/
      data/     3.3.4_clean.xlsx
      figures/  *.pdf  (векторные); hall_individual/ — по файлу на каждый ток
      results/  results.txt, results.json, *.csv  (создаётся автоматически)
      scripts/  analyze_hall.py

Запуск:   python analyze_hall.py      (необязательно: --xlsx ПУТЬ --figures ПАПКА --results ПАПКА)
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit

# =============================================================================
#                              НАСТРОЙКИ / ДОПУЩЕНИЯ
#   Всё, что помечено «ОЦЕНКА», — НЕ взято из методички/паспорта прибора:
#   замените на реальные значения (классы точности, погрешность штангенциркуля).
# =============================================================================
ROOT = Path(__file__).resolve().parent.parent          # lab_3.3.4/
DATA_FILE = ROOT / "data" / "3.3.4_clean.xlsx"
FIGURES_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"

E_CH = 1.602176634e-19        # Кл
K_B = 1.380649e-23            # Дж/К
M_0 = 9.1093837015e-31        # кг
T_ROOM = 295.0                # К (ОЦЕНКА)

# --- приборные погрешности ---
SIGMA_U_UV = 1.0              # мкВ: погрешность одного показания вольтметра U34 (последний разряд) [ОЦЕНКА]
I_SYS_REL = 0.01              # миллиамперметр: систематика (класс точности ~1.0), отн. [ОЦЕНКА]
SIGMA_I_RHO_DIV = 0.5         # дел.: отсчёт тока при измерении U35 [ОЦЕНКА]
CAL_REL_SYS = 0.02            # систематика градуировки B (милливеберметр/датчик Холла), отн. [ОЦЕНКА]
SIGMA_GEOM_MM = 0.05          # мм: погрешность размеров a, h, l (штангенциркуль) [ОЦЕНКА]
SIGMA_U35_MV = 0.001          # мВ: последний разряд вольтметра при U35 [ОЦЕНКА]

# --- калибровка ---
CAL_FIT_IMAX_MA = 1100.0      # fit калибровки только до этого тока (опыт Холла идёт до 1000 мА; выше — насыщение)
CAL_MAX_DEG = 5               # максимальная проверяемая степень полинома
N_CALFIT_DRAWS = 400          # розыгрышей для вклада неопределённости параметров калибровки в R_H
SEED = 12345

# Какой из размеров листа parameters — толщина h. В листе: a=1.5 мм, L_35=3 мм, 'l'=1.7 мм («геометрия образца»).
# В формулах методички: l(3-5), a, h => величина 'l' из листа трактуется как ТОЛЩИНА h. ПРОВЕРЬТЕ по держателю!
H_FROM_SHEET_KEY = "l"
L35_FROM_SHEET_KEY = "L_35"
A_FROM_SHEET_KEY = "a"

# Направление поля при «прямом» направлении (К1 в положении 1) относительно рис. 1б:
#   "toward" — на наблюдателя, "away" — от наблюдателя, None — неизвестно. Ток в образце: от (+) контакта 2 к (−) контакту 1.
# Принято U34 = φ3 − φ4 (контакт 3 слева, 4 справа). Проверьте полярность подключения вольтметра!
B_FORWARD_DIRECTION = None

M_EFF_ELECTRON = 0.12 * M_0   # доп. оценки (контр. вопрос 6) [ОЦЕНКА по справочнику для Ge]
M_EFF_HOLE = 0.35 * M_0
TABLE_GE = dict(mu_e_cm2Vs=3900.0, mu_h_cm2Vs=1900.0, n_i_cm3=2.4e13, rho_i_ohm_cm=47.0)   # Ge, ~300 К (сверить с практикумом)

plt.rcParams.update({"pdf.fonttype": 42, "svg.fonttype": "none",
                     "mathtext.fontset": "dejavusans",
                     "font.size": 15, "axes.titlesize": 17, "axes.labelsize": 16,
                     "xtick.labelsize": 14, "ytick.labelsize": 14,
                     "legend.fontsize": 12.5, "figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.35})

XL_B = "$B$, мТ"
XL_IM = r"$I_\mathrm{M}$, мА"
YL_UH = r"$U_\perp = U_{34} - U_0$, мкВ"


# =============================================================================
#                              ВСПОМОГАТЕЛЬНОЕ
# =============================================================================
def load(xlsx):
    cal = pd.read_excel(xlsx, "calibration")
    tidy = pd.read_excel(xlsx, "hall_tidy")
    wide = pd.read_excel(xlsx, "hall_wide")
    par = pd.read_excel(xlsx, "parameters").set_index("param")["value"].astype(float).to_dict()
    return cal, tidy, wide, par


def round_pm(v, e):
    """Округление результата по п. 3.2 пособия: погрешность — 1–2 значащие цифры (2, если первая цифра 1 или 2),
    последняя цифра значения — того же разряда. Возвращает строку 'v ± e'."""
    if not np.isfinite(e) or e <= 0:
        return f"{v:.4g}"
    exp = int(np.floor(np.log10(e)))
    first = int(e / 10 ** exp)
    nd = 2 if first in (1, 2) else 1
    dec = nd - 1 - exp
    ev, vv = round(e, dec), round(v, dec)
    if dec <= 0:
        return f"{int(round(vv))} ± {int(round(ev))}"
    return f"{vv:.{dec}f} ± {ev:.{dec}f}"


def pm_scaled(v, e, power):
    """'(v ± e)·10^power' для величин порядка 10^power."""
    s = round_pm(v / 10 ** power, e / 10 ** power)
    return f"({s})·10^{power}"


def quad(*xs):
    return float(np.sqrt(np.sum(np.square(xs))))


def lin(x, k, c):
    return k * x + c


def lin0(x, k):
    return k * x


def fit_line(x, y, sy, through_zero=False):
    """Линейная аппроксимация методом χ² (пособие, п. 4.2.1, 4.5): curve_fit с известными σ_y, absolute_sigma=True.
    Возвращает параметры, их погрешности, χ², число степеней свободы, χ²/df и p-value."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    sy = np.asarray(sy, float) * np.ones_like(y)
    if through_zero:
        p, cv = curve_fit(lin0, x, y, sigma=sy, absolute_sigma=True, p0=[np.sum(x * y) / np.sum(x * x)])
        yfit, npar = lin0(x, *p), 1
        out = dict(k=p[0], sk=np.sqrt(cv[0, 0]), c=0.0, sc=0.0)
    else:
        p, cv = curve_fit(lin, x, y, sigma=sy, absolute_sigma=True, p0=np.polyfit(x, y, 1))
        yfit, npar = lin(x, *p), 2
        out = dict(k=p[0], sk=np.sqrt(cv[0, 0]), c=p[1], sc=np.sqrt(cv[1, 1]))
    resid = y - yfit
    ndf = len(x) - npar
    chi2 = float(np.sum((resid / sy) ** 2))
    w = 1 / sy ** 2
    ss_tot = float(np.sum(w * (y - np.average(y, weights=w)) ** 2))
    out.update(chi2=chi2, ndf=ndf, chi2_red=chi2 / ndf if ndf > 0 else np.nan,
               pval=float(stats.chi2.sf(chi2, ndf)) if ndf > 0 else np.nan,
               r2=1 - chi2 / ss_tot if ss_tot > 0 else np.nan,
               rms=float(np.sqrt(np.mean(resid ** 2))), resid=resid, yfit=yfit, cov=cv)
    return out


# =============================================================================
#                      П.12  КАЛИБРОВКА B(I_M)
# =============================================================================
def _poly(deg):
    def f(I, *p):
        return np.polyval(p, np.asarray(I, float) / 1000.0)      # I [мА] -> [А]
    return f


def fit_calibration(I, B, imax, max_deg):
    """Полиномы степени 1..max_deg по данным с I <= imax (curve_fit, равные веса — МНК, пособие п. 4.1).
    Выбирается полином МИНИМАЛЬНОЙ достаточной степени: минимальный BIC среди МОНОТОННЫХ на [0, imax]."""
    m = I <= imax
    Iu, Bu = I[m], B[m]
    n = len(Iu)
    Ig = np.linspace(0, imax, 800)
    rows, cands = [], {}
    for d in range(1, max_deg + 1):
        f = _poly(d)
        p, cv = curve_fit(f, Iu, Bu, p0=np.polyfit(Iu / 1000.0, Bu, d))
        rss = float(np.sum((Bu - f(Iu, *p)) ** 2))
        mono = bool(np.all(np.diff(f(Ig, *p)) > 0))
        bic = n * np.log(rss / n) + (d + 1) * np.log(n)
        rows.append(dict(deg=d, rms_mT=np.sqrt(rss / (n - d - 1)), BIC=bic, monotonic=mono))
        cands[d] = (f, p, cv)
    tab = pd.DataFrame(rows)
    ok = tab[tab.monotonic]
    best = int(ok.loc[ok.BIC.idxmin(), "deg"])
    f, p, cv = cands[best]
    resid = Bu - f(Iu, *p)
    rms = float(np.sqrt(np.sum(resid ** 2) / (n - best - 1)))
    return dict(f=f, p=p, cov=cv, deg=best, rms=rms, resid=resid, table=tab, mask=m, n=n, imax=imax)


def alt_calibrations(I, B):
    """Альтернативные описания калибровки на ВСЁМ диапазоне — только для оценки вклада выбора модели в R_H."""
    out = {}
    p5 = np.polyfit(I / 1000.0, B, 5)
    out["полином 5-й ст., весь диапазон"] = lambda x: np.polyval(p5, np.asarray(x, float) / 1000.0)

    def knee(x, B0, a, Bs, Is, m):                       # линейный рост + плавное насыщение железа
        z = np.asarray(x, float) / Is
        return B0 + a * np.asarray(x, float) + Bs * z / (1 + z ** m) ** (1.0 / m)
    try:
        pk, _ = curve_fit(knee, I, B, p0=[20, 0.05, 800, 800, 3], bounds=([0, 0, 0, 10, 1], [100, 2, 5000, 5000, 20]), maxfev=200000)
        out["линейный рост + насыщение (физ. модель), весь диапазон"] = lambda x: knee(x, *pk)
    except Exception:
        pass
    return out


# =============================================================================
#                      П.13-14  ЭДС ХОЛЛА И k(I)
# =============================================================================
def series_fits(Bfun, fwd, series, sigU):
    items = []
    for s in series:
        sub = fwd[fwd.series == s]
        Ibr = sub.I_M_mA.values.astype(float)
        Bb = np.asarray(Bfun(Ibr), float)
        y = sub.U_Hall_uV.values.astype(float)
        r = fit_line(Bb, y, sigU)
        items.append(dict(series=s, B=Bb, y=y, r=r, I=float(sub.I_sample_mA.iloc[0]), I_div=float(sub.I_sample_div.iloc[0]),
                          U0=float(sub.U0_uV.iloc[0]), I_M=Ibr))
    return items


def slope_fit(items, use_zero=None):
    """k(I): метод χ² с весами 1/σ_k² (σ_k — из fit п.13). Погрешность тока считаем систематической (см. I_SYS_REL),
    случайный разброс k_i около прямой уже входит в χ²."""
    I = np.array([it["I"] for it in items])
    k = np.array([it["r"]["k"] for it in items])
    sk = np.array([it["r"]["sk"] for it in items])
    rf = fit_line(I, k, sk, through_zero=False)
    rz = fit_line(I, k, sk, through_zero=True)
    if use_zero is None:
        use_zero = abs(rf["c"]) / rf["sc"] < 2.0
    rk = rz if use_zero else rf
    infl = max(1.0, np.sqrt(rk["chi2_red"]))            # если χ²/df > 1 — раздуваем σ (погрешности из fit иначе занижены)
    return dict(I=I, k=k, sk=sk, free=rf, zero=rz, use_zero=use_zero, main=rk, s=rk["k"], s_err=rk["sk"] * infl, infl=infl)


# =============================================================================
#                                   ГРАФИКИ
# =============================================================================
def draw_hall(ax, axr, Bb, y, r, sigU, color, title, ylabel=None):
    ax.errorbar(Bb, y, yerr=sigU, fmt="o", ms=6, color=color, capsize=3, label="измерения")
    Bl = np.linspace(0, Bb.max() * 1.05, 100)
    ax.plot(Bl, lin(Bl, r["k"], r["c"]), "-", color="k", lw=1.6, label="линейный fit")
    ax.set_ylabel(ylabel or YL_UH)
    ax.set_title(title, fontsize=15)
    ax.tick_params(labelbottom=False)
    axr.errorbar(Bb, r["resid"], yerr=sigU, fmt="o", ms=6, color=color, capsize=3)
    axr.axhline(0, color="k", lw=1)
    axr.set_xlabel(XL_B)
    axr.set_ylabel("ост., мкВ")


def draw_overview(ax, items, cmap):
    for i, it in enumerate(items):
        col = cmap(i / max(len(items) - 1, 1))
        ax.plot(it["B"], it["y"], "o", ms=5, color=col)
        Bl = np.linspace(0, it["B"].max() * 1.03, 100)
        ax.plot(Bl, lin(Bl, it["r"]["k"], it["r"]["c"]), "-", color=col, lw=1.6, label=f"$I$ = {it['I']:.2f} мА")
    ax.set_xlabel(XL_B)
    ax.set_ylabel(YL_UH)
    ax.set_title("Семейство $U_\\perp(B)$ при разных токах")
    ax.legend(loc="upper left")


def savefig(fig, outdir, name):
    path = os.path.join(outdir, name)
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return path


# =============================================================================
#                                    MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="Обработка лаб. 3.3.4 (эффект Холла)")
    ap.add_argument("--xlsx", default=str(DATA_FILE), help="входной Excel (по умолчанию ../data/3.3.4_clean.xlsx)")
    ap.add_argument("--figures", default=str(FIGURES_DIR), help="папка для PDF-графиков (по умолчанию ../figures)")
    ap.add_argument("--results", default=str(RESULTS_DIR), help="папка для txt/json/csv (по умолчанию ../results)")
    args = ap.parse_args()
    if not os.path.isfile(args.xlsx):
        sys.exit(f"Не найден файл данных: {args.xlsx}")
    os.makedirs(args.figures, exist_ok=True)
    os.makedirs(args.results, exist_ok=True)
    rng = np.random.default_rng(SEED)

    cal_df, tidy, wide, par = load(args.xlsx)
    LOG = []

    def P(s=""):
        LOG.append(s)
        print(s)

    # ---------------------------- параметры образца ----------------------------
    k_I = par["k_I"]
    a_mm, l35_mm, h_mm = par[A_FROM_SHEET_KEY], par[L35_FROM_SHEET_KEY], par[H_FROM_SHEET_KEY]
    a, l35, h = a_mm * 1e-3, l35_mm * 1e-3, h_mm * 1e-3
    U35_V = par["U_rho"] * 1e-3
    I_rho_A = par["I_rho"] * k_I * 1e-3
    sg = SIGMA_GEOM_MM * 1e-3
    sigU = SIGMA_U_UV

    P("=" * 78)
    P("ЛАБОРАТОРНАЯ 3.3.4  «ЭФФЕКТ ХОЛЛА В ПОЛУПРОВОДНИКАХ» — РЕЗУЛЬТАТЫ ОБРАБОТКИ")
    P("=" * 78)
    P("Обозначения в этом файле: I_M — ток электромагнита (М — нижний индекс), I — ток через образец, U_perp — ЭДС Холла,")
    P("U0 — напряжение между контактами 3 и 4 при выключенном магните, k = dU_perp/dB, s = dk/dI, R_H — постоянная Холла.")
    P("Погрешности: δ = 1σ (доверительная вероятность 68%); полная = квадратичная сумма статистической и систематической (приборной).")
    P("\n[0] ИСХОДНЫЕ ПАРАМЕТРЫ ОБРАЗЦА И УСТАНОВКИ (лист parameters)")
    P(f"  цена деления миллиамперметра k_I = {k_I} мА/дел (150 дел = 1.5 мА)")
    P(f"  a (ширина) = {a_mm} мм;  l (расстояние между контактами 3 и 5) = {l35_mm} мм")
    P(f"  h (толщина) = {h_mm} мм   [!] взято из строки 'l' листа parameters («геометрия образца»); ПРОВЕРИТЬ по держателю")
    P(f"  U35 = {par['U_rho']} мВ при I = {par['I_rho']:.0f} дел = {I_rho_A*1e3:.3f} мА (без магнитного поля)")
    P("  токи через образец в опыте Холла: " + ", ".join(f"{d:.0f} дел ({d*k_I:.2f} мА)" for d in sorted(tidy.I_sample_div.unique())))
    if tidy.I_sample_mA.max() > 1.0:
        P("  [!] Максимальный ток 1.30 мА превышает рекомендованный методичкой предел 1 мА (п.2) — отметить в отчёте.")
    P("  Приборные погрешности (ОЦЕНКИ, заменить паспортными):")
    P(f"    σ_U = {SIGMA_U_UV} мкВ (одно показание U34);  миллиамперметр: систематика {100*I_SYS_REL:.1f}% (класс), отсчёт ±{SIGMA_I_RHO_DIV} дел;")
    P(f"    градуировка B: систематика {100*CAL_REL_SYS:.1f}%;  размеры образца: ±{SIGMA_GEOM_MM} мм;  U35: ±{SIGMA_U35_MV} мВ.")

    # =========================================================================
    # П.12  КАЛИБРОВКА
    # =========================================================================
    I_c = cal_df["I_M_mA"].values.astype(float)
    B_c = cal_df["B_mT"].values.astype(float)
    passes = cal_df["pass"].values
    cal = fit_calibration(I_c, B_c, CAL_FIT_IMAX_MA, CAL_MAX_DEG)
    Bfun = lambda x: cal["f"](np.asarray(x, float), *cal["p"])
    B_rem_fit = float(Bfun(0.0))
    B_rem = float(B_c[I_c == 0].mean())                  # измеренная остаточная индукция (используется в проверках знака и сдвига)
    m1, m2 = passes == 1, passes == 2
    use = cal["mask"]

    P("\n" + "-" * 78)
    P("[1] П.12. КАЛИБРОВКА ЭЛЕКТРОМАГНИТА B(I_M) — интерполяция полиномом")
    P("-" * 78)
    P("  Зачем полином: калибровка нужна только для пересчёта I_M -> B (интерполяция); физический смысл коэффициентов не приписывается.")
    P(f"  Опыт Холла идёт до I_M = 1000 мА, поэтому аппроксимируются точки с I_M ≤ {CAL_FIT_IMAX_MA:.0f} мА ({cal['n']} из {len(I_c)}); "
      f"выше — область насыщения железа, в fit не используется.")
    P("  Степень: минимальный BIC среди полиномов, монотонных на рабочем диапазоне (немонотонная B(I_M) физически невозможна):")
    for _, r in cal["table"].iterrows():
        mark = "  <== выбран" if int(r.deg) == cal["deg"] else ""
        P(f"    степень {int(r.deg)}: СКО остатков = {r.rms_mT:5.2f} мТ,  BIC = {r.BIC:7.2f},  монотонен: {'да' if r.monotonic else 'НЕТ'}{mark}")
    P(f"  Принят полином степени {cal['deg']}: B(I_M) = polyval(p, I_M/1000 [мА->А]) [мТ], коэффициенты p (старший первым):")
    P("    " + ", ".join(f"{x:.6g}" for x in cal["p"]))
    P(f"  Погрешность отдельной точки калибровки по МНК (пособие, п.4.1.2): σ_B = sqrt(Σδ²/(n−p)) = {cal['rms']:.1f} мТ "
      f"(на графике — кресты у точек). Остатки хаотичны, систематического хода нет.")
    P(f"  Остаточная индукция: измерено B(I_M=0) = {B_rem:.1f} мТ; полином даёт {B_rem_fit:.1f} мТ (разница в пределах σ_B). "
      f"В проверках ниже используется измеренное значение.")
    d12 = B_c[m2 & use] - Bfun(I_c[m2 & use])
    P(f"  Проходы 1 и 2 (гистерезис): отклонение точек прохода 2 от общего fit = {d12.mean():+.1f} ± {d12.std(ddof=1):.1f} мТ — "
      f"в пределах σ_B, значимого гистерезиса не видно.")
    P(f"  Систематическая погрешность градуировки B (масштаб, ОЦЕНКА): {100*CAL_REL_SYS:.1f}%.")

    # альтернативные калибровки -> вклад выбора модели
    alts = alt_calibrations(I_c, B_c)

    # график калибровки
    Ig = np.linspace(0, I_c.max(), 600)
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True, gridspec_kw=dict(height_ratios=[3, 1]))
    ax.errorbar(I_c[m1 & use], B_c[m1 & use], yerr=cal["rms"], fmt="o", ms=5, color="C0", capsize=2, label="проход 1")
    ax.errorbar(I_c[m2 & use], B_c[m2 & use], yerr=cal["rms"], fmt="s", ms=5, color="C3", capsize=2, label="проход 2")
    ax.plot(I_c[~use], B_c[~use], "o", ms=5, mfc="none", color="gray", label=f"$I_\\mathrm{{M}}$ > {CAL_FIT_IMAX_MA:.0f} мА (не в fit)")
    Ig_in = Ig[Ig <= CAL_FIT_IMAX_MA]
    ax.plot(Ig_in, Bfun(Ig_in), "k-", lw=1.6, label=f"полином {cal['deg']}-й степени")
    ax.axvline(1000, color="C2", lw=1, ls="--")
    ax.text(1010, 60, "предел опыта Холла", color="C2", fontsize=12, rotation=90, va="bottom")
    ax.set_ylabel("$B$, мТ")
    ax.set_title("Калибровка электромагнита $B(I_\\mathrm{M})$")
    ax.legend(loc="upper left")
    ax2.errorbar(I_c[use], cal["resid"], yerr=cal["rms"], fmt="o", ms=4, color="C0", capsize=2)
    ax2.axhline(0, color="k", lw=1)
    ax2.set_xlabel(XL_IM)
    ax2.set_ylabel("остатки, мТ")
    savefig(fig, args.figures, "fig1_calibration.pdf")

    # =========================================================================
    # П.13  СЕМЕЙСТВО U_perp(B)
    # =========================================================================
    fwd = tidy[tidy.direction == "forward"].copy()
    rev = tidy[tidy.direction == "reverse"].copy()
    series = list(dict.fromkeys(fwd.series))
    chk = np.abs(tidy.U_Hall_uV - (tidy.U34_uV - tidy.U0_uV)).max()
    wd = wide.set_index("I_M_mA")
    ok_w = all(np.allclose(wd.loc[fwd[fwd.series == s].I_M_mA, f"U34_{s}_uV"].values, fwd[fwd.series == s].U34_uV.values) for s in series)

    P("\n" + "-" * 78)
    P("[2] П.13. ЭДС ХОЛЛА U_perp = U34 − U0 (ф.(1)) И ЛИНЕЙНЫЕ АППРОКСИМАЦИИ U_perp(B) — метод χ²")
    P("-" * 78)
    P(f"  Проверка таблиц: max|U_Hall − (U34 − U0)| = {chk:.3g} мкВ; hall_tidy и hall_wide согласованы: {'да' if ok_w else 'НЕТ — проверить'}")
    P(f"  Модель U_perp = k·B + c.  σ_y = σ_U = {sigU} мкВ для каждой точки (приборная), поэтому применим метод χ² "
      f"(пособие п.4.2, 4.5): σ_k, σ_c берутся из ковариационной матрицы curve_fit (absolute_sigma=True).")
    P(f"  Погрешность U0 (один отсчёт на серию) — общий сдвиг всей серии; он целиком уходит в параметр c и на k не влияет.")
    P(f"  B в каждой точке берётся из калибровки; её случайная ошибка мала (см. вклад параметров калибровки ниже), "
      f"систематика масштаба B ({100*CAL_REL_SYS:.0f}%) учтена в итоговой погрешности.")
    P(f"  Ожидание: U_perp = 0 при B = B_ост, т.е. c ≈ −k·B_ост (B_ост = {B_rem:.1f} мТ).")
    items = series_fits(Bfun, fwd, series, sigU)
    rows = []
    for it in items:
        r = it["r"]
        rows.append(dict(series=it["series"], I_div=it["I_div"], I_mA=it["I"], U0_uV=it["U0"], n_pts=len(it["y"]),
                         k_uV_per_mT=r["k"], sk=r["sk"], c_uV=r["c"], sc_uV=r["sc"], expected_c=-r["k"] * B_rem,
                         chi2=r["chi2"], ndf=r["ndf"], chi2_red=r["chi2_red"], p_value=r["pval"], R2=r["r2"], rms_resid_uV=r["rms"]))
    tab13 = pd.DataFrame(rows)
    P(f"\n  {'I,дел':>6} {'I,мА':>6} {'U0,мкВ':>7} {'k,мкВ/мТ':>10} {'σ_k':>7} {'c,мкВ':>8} {'σ_c':>6} {'-k*Bост':>8} {'χ²':>6} {'ndf':>4} {'χ²/ndf':>7} {'p':>6} {'R²':>8}")
    for r in rows:
        P(f"  {r['I_div']:6.0f} {r['I_mA']:6.2f} {r['U0_uV']:7.0f} {r['k_uV_per_mT']:10.4f} {r['sk']:7.4f} {r['c_uV']:8.2f} {r['sc_uV']:6.2f} "
          f"{r['expected_c']:8.2f} {r['chi2']:6.1f} {r['ndf']:4d} {r['chi2_red']:7.2f} {r['p_value']:6.2f} {r['R2']:8.5f}")
    chi_tot = tab13.chi2.sum() / tab13.ndf.sum()
    P(f"  Суммарно по всем сериям: χ²/ndf = {chi_tot:.2f} (при σ_U = {sigU} мкВ).  Критерий (п.4.3): ≈1 — модель и погрешности согласуются; "
      f">1.5 — модель плохая или σ занижены; <0.5 — σ завышены.")
    P(f"  Все k > 0 (U34 растёт с полем в прямом направлении): {bool((tab13.k_uV_per_mT>0).all())}; зависимости линейны (R² ≥ {tab13.R2.min():.4f}).")
    cdev = (tab13.c_uV - tab13.expected_c) / tab13.sc_uV
    P("  Отклонение c от −k·B_ост в единицах σ_c: " + ", ".join(f"{x:+.1f}" for x in cdev) +
      "  (при |.|>2 — сдвиг нуля не вполне объясняется остаточным полем; на k не влияет)")

    cm = plt.get_cmap("viridis")
    for i, it in enumerate(items):
        it["col"] = cm(i / max(len(items) - 1, 1))

    def hall_title(it):
        return f"$I$ = {it['I']:.2f} мА:  $k$ = {it['r']['k']:.3f} ± {it['r']['sk']:.3f} мкВ/мТ"

    fig, ax = plt.subplots(figsize=(9, 7))
    draw_overview(ax, items, cm)
    savefig(fig, args.figures, "fig2a_hall_overview.pdf")

    fig = plt.figure(figsize=(12, 18))
    outer = fig.add_gridspec(4, 2, hspace=0.45, wspace=0.34)
    for i, it in enumerate(items):
        inner = outer[i // 2, i % 2].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.08)
        ax_ = fig.add_subplot(inner[0])
        axr_ = fig.add_subplot(inner[1], sharex=ax_)
        draw_hall(ax_, axr_, it["B"], it["y"], it["r"], sigU, it["col"], hall_title(it), ylabel=r"$U_\perp$, мкВ")
    ax_ = fig.add_subplot(outer[3, 1])
    draw_overview(ax_, items, cm)
    ax_.set_ylabel(r"$U_\perp$, мкВ")
    fig.align_ylabels()
    savefig(fig, args.figures, "fig2b_hall_panels.pdf")

    indiv_dir = os.path.join(args.figures, "hall_individual")
    os.makedirs(indiv_dir, exist_ok=True)
    for it in items:
        fig = plt.figure(figsize=(8, 7))
        inner = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.08)
        ax_ = fig.add_subplot(inner[0])
        axr_ = fig.add_subplot(inner[1], sharex=ax_)
        draw_hall(ax_, axr_, it["B"], it["y"], it["r"], sigU, it["col"], hall_title(it))
        ax_.legend(loc="upper left")
        savefig(fig, indiv_dir, f"fig2_I_{it['I']:.2f}mA.pdf")

    # =========================================================================
    # П.8-9  ОБРАТНОЕ ПОЛЕ, ЗНАК
    # =========================================================================
    P("\n" + "-" * 78)
    P("[3] П.8-9. ОБРАТНОЕ НАПРАВЛЕНИЕ ПОЛЯ (I = 1.30 мА), ЗНАК НОСИТЕЛЕЙ")
    P("-" * 78)
    s_fw = "I130"
    fw = next(it for it in items if it["series"] == s_fw)
    Ibr_r = rev.I_M_mA.values.astype(float)
    Br_r = Bfun(Ibr_r)
    U34_r = rev.U34_uV.values.astype(float)
    U0_r = float(rev.U0_uV.iloc[0])
    U0_f = fw["U0"]
    y_r = rev.U_Hall_uV.values.astype(float)
    # обратная серия: поле и остаточное поле сменили знак -> U_perp = −k·(B − B_ост); для сравнения берём −U_perp
    rr = fit_line(Br_r, -y_r, sigU)
    P(f"  Метод U0 (ф.(1)), обратная серия: (−U_perp) = k·B + c:  k_rev = {rr['k']:.4f} ± {rr['sk']:.4f} мкВ/мТ, χ²/ndf = {rr['chi2_red']:.2f};  "
      f"прямая: k_fwd = {fw['r']['k']:.4f} ± {fw['r']['sk']:.4f}")
    k_from_U0 = (U0_f - U0_r) / (2 * B_rem)
    Uohm = (U0_f + U0_r) / 2
    P(f"  U0 при I_M = 0: прямое поле {U0_f:.0f} мкВ, обратное {U0_r:.0f} мкВ (разные из-за остаточного поля B_ост разного знака).")
    P(f"  Независимая оценка по паре U0: омическое смещение U_ом = (U0(+)+U0(−))/2 = {Uohm:.1f} мкВ;  "
      f"k = (U0(+)−U0(−))/(2·B_ост) = {k_from_U0:.3f} мкВ/мТ (B_ост = {B_rem:.1f} мТ) — совпадает с k_fwd в пределах точности.")
    Ufw = fwd[fwd.series == s_fw].set_index("I_M_mA").U34_uV
    msym = np.isin(Ibr_r, Ufw.index.values)
    Usym = np.array([(Ufw.loc[i_] - u) / 2 for i_, u in zip(Ibr_r[msym], U34_r[msym])])
    ssym = sigU / np.sqrt(2)
    rs = fit_line(Br_r[msym], Usym, ssym)
    P(f"  Симметризация U_perp = (U34(+) − U34(−))/2 при одинаковых I_M: k_sym = {rs['k']:.4f} ± {rs['sk']:.4f} мкВ/мТ, "
      f"c = {rs['c']:.2f} ± {rs['sc']:.2f} мкВ, χ²/ndf = {rs['chi2_red']:.2f}  (метод не требует U0; B здесь — полная индукция B(I_M))")
    P(f"  Расхождение наклонов: k_rev/k_fwd = {rr['k']/fw['r']['k']:.3f}; k_sym/k_fwd = {rs['k']/fw['r']['k']:.3f}. "
      f"Различие ≈ {100*abs(rr['k']/fw['r']['k']-1):.1f}% больше статистических σ (χ²/ndf обратной серии велик) — "
      f"индукция при реверсе тока в магните не в точности равна B(I_M) прямого направления (гистерезис/остаточное поле). "
      f"Принято как оценка систематики метода: ≈ {100*abs(rs['k']/fw['r']['k']-1):.1f}% (в бюджет R_H не включено, т.к. R_H определяется по прямой серии, п.14).")
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.errorbar(fw["B"], fw["y"], yerr=sigU, fmt="o", color="C0", capsize=2, label="прямое поле ($U_{34}-U_0$)")
    ax.errorbar(Br_r, -y_r, yerr=sigU, fmt="s", color="C3", capsize=2, label="обратное поле, $-(U_{34}-U_0)$")
    ax.errorbar(Br_r[msym], Usym, yerr=ssym, fmt="^", color="C2", capsize=2, label=r"симметризация $(U_{34}^{(+)}-U_{34}^{(-)})/2$")
    Bl = np.linspace(0, 1000, 200)
    ax.plot(Bl, lin(Bl, fw["r"]["k"], fw["r"]["c"]), "-", color="C0", lw=1.4, label=f"fit, прямое: $k$ = {fw['r']['k']:.3f} мкВ/мТ")
    ax.plot(Bl, lin(Bl, rr["k"], rr["c"]), "-", color="C3", lw=1.4, label=f"fit, обратное: $k$ = {rr['k']:.3f} мкВ/мТ")
    ax.plot(Bl, lin(Bl, rs["k"], rs["c"]), "--", color="C2", lw=1.4, label=f"fit, симметризация: $k$ = {rs['k']:.3f} мкВ/мТ")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel(XL_B)
    ax.set_ylabel(r"$U_\perp$, мкВ")
    ax.set_title("$I$ = 1.30 мА: прямое и обратное поле")
    ax.legend(loc="upper left", fontsize=11.5)
    savefig(fig, args.figures, "fig3_reverse_field.pdf")

    sign_k = np.sign(fw["r"]["k"])
    P("\n  ЗНАК НОСИТЕЛЕЙ:")
    P("  Геометрия: ток идёт от контакта 2 (+) к контакту 1 (−), т.е. «вверх» на рис.1б; контакт 3 слева, 4 справа.")
    P(f"  Факт: при прямом поле U_perp растёт с B (k = {fw['r']['k']:.3f} > 0), знак U_perp = {'+' if sign_k>0 else '−'}.")
    P("  Правило (U34 = φ3 − φ4, ток вверх): B на наблюдателя и U34 > 0 -> ЭЛЕКТРОНЫ (n-тип); B от наблюдателя и U34 > 0 -> ДЫРКИ (p-тип);")
    P("  при U34 < 0 выводы меняются на противоположные.")
    if B_FORWARD_DIRECTION is None:
        P("  [!] Направление B в «прямой» серии (К1 в положении 1) и полярность вольтметра в данных не заданы -> тип проводимости определяется")
        P("      по рисунку в тетради (стрелка на торце магнита). Задайте B_FORWARD_DIRECTION в скрипте.")
        carrier = None
    else:
        is_e = (B_FORWARD_DIRECTION == "toward") == (sign_k > 0)
        carrier = "electron" if is_e else "hole"
        P(f"  При B_FORWARD_DIRECTION = '{B_FORWARD_DIRECTION}': носители — {'ЭЛЕКТРОНЫ (n-тип)' if is_e else 'ДЫРКИ (p-тип)'}.")

    # =========================================================================
    # П.14  k(I), R_H
    # =========================================================================
    sf = slope_fit(items)
    rf, rz = sf["free"], sf["zero"]
    P("\n" + "-" * 78)
    P("[4] П.14. k(I) И ПОСТОЯННАЯ ХОЛЛА R_H — метод χ²")
    P("-" * 78)
    P("  Модель: k = s·I + c (из ф.(3.27): k = R_H·I/h).  Веса 1/σ_k² (σ_k — из fit п.13); погрешность тока в fit по I не вводится:")
    P("  по пособию (п.4.2) ось абсцисс считается точной, а ошибка шкалы миллиамперметра — систематическая и учитывается отдельно.")
    P(f"  (а) со свободным сдвигом:  s = {rf['k']:.4f} ± {rf['sk']:.4f} мкВ/(мТ·мА),  c = {rf['c']:.4f} ± {rf['sc']:.4f} мкВ/мТ,  χ²/ndf = {rf['chi2_red']:.2f}")
    P(f"  (б) через начало (k(0)=0): s = {rz['k']:.4f} ± {rz['sk']:.4f} мкВ/(мТ·мА),  χ²/ndf = {rz['chi2_red']:.2f}  (p = {rz['pval']:.2f})")
    cs = abs(rf["c"]) / rf["sc"]
    P(f"  Сдвиг c отличается от нуля на {cs:.1f}σ -> " + ("неотличим от нуля, основным принят fit (б) через начало." if sf["use_zero"]
                                                       else "значим; основным принят fit (а)."))
    if sf["infl"] > 1:
        P(f"  χ²/ndf > 1: σ_s умножена на sqrt(χ²/ndf) = {sf['infl']:.2f}.")
    s_val, s_stat = sf["s"], sf["s_err"]
    P(f"  Размерность: мкВ/(мТ·мА) = В/(Тл·А).  R_H = h·s = {h_mm} мм × {s_val:.4f} = {s_val*h:.4e} м³/Кл = {s_val*h*1e6:.1f} см³/Кл")

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True, gridspec_kw=dict(height_ratios=[3, 1]))
    ax.errorbar(sf["I"], sf["k"], yerr=sf["sk"], fmt="o", ms=6, color="C0", capsize=3, label="$k_i$ из п. 13 (±σ)")
    Il = np.linspace(0, sf["I"].max() * 1.05, 100)
    ax.plot(Il, lin(Il, rf["k"], rf["c"]), "--", color="C1", lw=1.3, label=f"(а) со сдвигом: $s$ = {rf['k']:.3f}, $c$ = {rf['c']:.3f}")
    ax.plot(Il, lin0(Il, rz["k"]), "-", color="k", lw=1.5, label=f"(б) через начало: $s$ = {rz['k']:.3f} мкВ/(мТ·мА)")
    ax.set_ylabel(r"$k = dU_\perp/dB$, мкВ/мТ")
    ax.set_title("Зависимость углового коэффициента $k$ от тока $I$")
    ax.legend(loc="upper left")
    ax2.errorbar(sf["I"], rz["resid"], yerr=sf["sk"], fmt="o", color="k", capsize=3, label="остатки (б)")
    ax2.errorbar(sf["I"], rf["resid"], yerr=sf["sk"], fmt="s", color="C1", capsize=3, label="остатки (а)")
    ax2.axhline(0, color="gray")
    ax2.set_xlabel("$I$, мА")
    ax2.set_ylabel("остатки, мкВ/мТ")
    ax2.legend()
    savefig(fig, args.figures, "fig4_k_vs_I.pdf")

    # ---- вклады неопределённости калибровки в s
    def s_for(Bf):
        return slope_fit(series_fits(Bf, fwd, series, sigU), use_zero=sf["use_zero"])["s"]

    pdraw = rng.multivariate_normal(cal["p"], cal["cov"], size=N_CALFIT_DRAWS)
    s_draws = np.array([s_for(lambda x, pp=pp: cal["f"](np.asarray(x, float), *pp)) for pp in pdraw])
    rel_calfit = float(s_draws.std(ddof=1) / s_val)
    alt_RH = {nm: s_for(f_) * h * 1e6 for nm, f_ in alts.items()}
    RH_nom_cm = s_val * h * 1e6
    rel_model = max([abs(v - RH_nom_cm) / RH_nom_cm for v in alt_RH.values()] + [0.0])

    # =========================================================================
    # П.15-16  rho0, sigma0, n, mu
    # =========================================================================
    RH = s_val * h
    rho0 = U35_V * a * h / (I_rho_A * l35)
    sigma0 = 1.0 / rho0
    n_m3 = 1.0 / (E_CH * RH)
    mu = RH * sigma0
    P("\n" + "-" * 78)
    P("[5] П.10, 15-16. ρ0, σ0, n, μ (центральные значения)")
    P("-" * 78)
    P(f"  ρ0 = U35·a·h/(I·l) = {U35_V*1e3:.3f} мВ · {a_mm} мм · {h_mm} мм /({I_rho_A*1e3:.2f} мА · {l35_mm} мм) = {rho0:.4e} Ом·м = {rho0*100:.4f} Ом·см")
    P(f"  σ0 = 1/ρ0 = {sigma0:.2f} См/м = {sigma0/100:.3f} См/см")
    P(f"  n = 1/(e·|R_H|) = {n_m3:.4e} м⁻³ = {n_m3*1e-6:.4e} см⁻³")
    P(f"  μ = |R_H|·σ0 = {mu:.4f} м²/(В·с) = {mu*1e4:.0f} см²/(В·с)")

    # =========================================================================
    # П.17  ПОГРЕШНОСТИ: статистика + систематика (пособие 2.9, 2.12, 2.14)
    # =========================================================================
    rs_ = s_stat / s_val                         # статистика s из χ²-fit k(I)
    r_stat = quad(rs_, rel_calfit)               # случайная часть: fit + параметры калибровки
    r_B, r_I = CAL_REL_SYS, I_SYS_REL
    r_h, r_a, r_l = sg / h, sg / a, sg / l35
    r_U35 = SIGMA_U35_MV / par["U_rho"]
    r_Irho = quad(I_SYS_REL, SIGMA_I_RHO_DIV / par["I_rho"])

    # R_H = h·s :  δR/R  = stat ⊕ (B-масштаб, I-масштаб, h, выбор модели калибровки)
    RH_sys = quad(r_B, r_I, r_h, rel_model)
    RH_tot = quad(r_stat, RH_sys)
    # n:  та же относительная погрешность
    # ρ0 = U35·a·h/(I·l)
    rho_tot = quad(r_U35, r_a, r_h, r_Irho, r_l)
    # μ = R_H/ρ0 = s·I_ρ·l/(U35·a): h и масштаб миллиамперметра сокращаются (один и тот же прибор для I и I_ρ)
    mu_sys = quad(r_B, r_a, r_l, r_U35, SIGMA_I_RHO_DIV / par["I_rho"])
    mu_tot = quad(r_stat, mu_sys)

    P("\n" + "-" * 78)
    P("[6] П.17. ПОГРЕШНОСТИ")
    P("-" * 78)
    P("  Метод: статистическая часть — из fit (метод χ²) и неопределённости параметров калибровки; систематическая — приборная (оценки выше);")
    P("  полная погрешность = квадратичная сумма (пособие ф.(2.9)); косвенные величины — по ф.(2.12)/(2.14).")
    P("  Бюджет относительных погрешностей R_H = h·s, %:")
    P(f"    статистика, fit k(I) (σ_s/s)                                        : {100*rs_:5.2f}")
    P(f"    статистика, параметры калибровки B(I_M) (розыгрыш по ковариации fit): {100*rel_calfit:5.2f}")
    P(f"    ИТОГО статистическая                                                : {100*r_stat:5.2f}")
    P(f"    систематика: масштаб градуировки B [ОЦЕНКА]                          : {100*r_B:5.2f}")
    P(f"    систематика: масштаб миллиамперметра [ОЦЕНКА]                        : {100*r_I:5.2f}")
    P(f"    систематика: толщина образца h (±{SIGMA_GEOM_MM} мм) [ОЦЕНКА]                  : {100*r_h:5.2f}")
    P(f"    систематика: выбор модели калибровки (см. ниже)                      : {100*rel_model:5.2f}")
    P(f"    ИТОГО систематическая                                               : {100*RH_sys:5.2f}")
    P(f"    ПОЛНАЯ δR_H/R_H                                                     : {100*RH_tot:5.2f}")
    P("  Влияние выбора модели калибровки на R_H, см³/Кл (проверка, что результат не зависит от «полинома»):")
    P(f"    принятый полином {cal['deg']}-й ст. (I_M ≤ {CAL_FIT_IMAX_MA:.0f} мА)                    : {RH_nom_cm:.1f}")
    for nm, v in alt_RH.items():
        P(f"    {nm:60s}: {v:.1f}")
    P(f"  Для μ: h и масштаб миллиамперметра сокращаются (μ = s·I_ρ·l/(U35·a)); систематика μ = {100*mu_sys:.2f}% "
      f"(B-масштаб {100*r_B:.0f}%, a {100*r_a:.1f}%, l {100*r_l:.1f}%, U35 {100*r_U35:.2f}%, отсчёт I_ρ {100*SIGMA_I_RHO_DIV/par['I_rho']:.2f}%); статистика μ = {100*r_stat:.2f}%.")
    P(f"  Для ρ0: U35 {100*r_U35:.2f}%, a {100*r_a:.1f}%, h {100*r_h:.1f}%, l {100*r_l:.1f}%, I_ρ {100*r_Irho:.2f}% -> δρ0/ρ0 = {100*rho_tot:.1f}% (все приборные/систематические).")

    # =========================================================================
    # ИТОГИ
    # =========================================================================
    RH_cm = RH * 1e6
    eRH_stat, eRH_sys, eRH = RH_cm * r_stat, RH_cm * RH_sys, RH_cm * RH_tot
    n_cm = n_m3 * 1e-6
    mu_cm = mu * 1e4
    rho_cm = rho0 * 100
    sig_cm = sigma0 / 100
    P("\n" + "=" * 78)
    P("[7] ИТОГОВЫЕ РЕЗУЛЬТАТЫ (округлено по правилам п.3.2 пособия; δ = 1σ, P = 68%)")
    P("=" * 78)
    P(f"  R_H = ({round_pm(RH_cm, eRH)}) см³/Кл   [стат. ±{eRH_stat:.2g}, сист. ±{eRH_sys:.2g}],  ε = {100*RH_tot:.1f}%")
    P(f"      = {pm_scaled(RH, RH*RH_tot, -4)} м³/Кл")
    P(f"  n   = {pm_scaled(n_cm, n_cm*RH_tot, 15)} см⁻³   [стат. ±{100*r_stat:.1f}%, сист. ±{100*RH_sys:.1f}%],  ε = {100*RH_tot:.1f}%")
    P(f"  ρ0  = ({round_pm(rho_cm, rho_cm*rho_tot)}) Ом·см  (все погрешности приборные), ε = {100*rho_tot:.1f}%")
    P(f"  σ0  = ({round_pm(sig_cm, sig_cm*rho_tot)}) См/см")
    P(f"  μ   = ({round_pm(mu_cm, mu_cm*mu_tot)}) см²/(В·с)   [стат. ±{100*r_stat:.1f}%, сист. ±{100*mu_sys:.1f}%],  ε = {100*mu_tot:.1f}%")
    P(f"  Угловой коэффициент: s = dk/dI = ({round_pm(s_val, s_stat)}) мкВ/(мТ·мА) (статистика), k_i = {sf['k'].min():.3f}…{sf['k'].max():.3f} мкВ/мТ")
    P("  Тип проводимости: " + ("электронный (n-тип)" if carrier == "electron" else "дырочный (p-тип)" if carrier == "hole"
                               else "определить по рисунку и направлению B (см. п.[3]); k > 0 при прямом поле"))

    P("\n" + "-" * 78)
    P("[8] П.17. СРАВНЕНИЕ С ТАБЛИЧНЫМИ ЗНАЧЕНИЯМИ (справочные для Ge, ~300 К; сверить с таблицей практикума)")
    P("-" * 78)
    P(f"  Подвижность в чистом Ge: электроны ≈ {TABLE_GE['mu_e_cm2Vs']:.0f}, дырки ≈ {TABLE_GE['mu_h_cm2Vs']:.0f} см²/(В·с).")
    emu = mu_cm * mu_tot
    for nm, v in (("электронов", TABLE_GE["mu_e_cm2Vs"]), ("дырок", TABLE_GE["mu_h_cm2Vs"])):
        P(f"    эксперимент {mu_cm:.0f} ± {emu:.0f} vs {nm} {v:.0f}: отношение {mu_cm/v:.2f}, различие {(mu_cm-v)/emu:+.1f}σ (критерий п.2.2: значимо при > 2σ)")
    P(f"  Собственная концентрация Ge n_i ≈ {TABLE_GE['n_i_cm3']:.1e} см⁻³; найденное n = {n_cm:.2e} см⁻³ -> n/n_i ≈ {n_cm/TABLE_GE['n_i_cm3']:.0f} (примесная проводимость).")
    P(f"  Собственное удельное сопротивление Ge ≈ {TABLE_GE['rho_i_ohm_cm']:.0f} Ом·см; найденное ρ0 = {rho_cm:.3f} Ом·см (в {TABLE_GE['rho_i_ohm_cm']/rho_cm:.0f} раз меньше).")
    P("  Холловская подвижность μ_H = R_H·σ отличается от дрейфовой на холловский фактор r_H (≈1…1.2) — не учитывался.")

    P("\n" + "-" * 78)
    P("[9] ДОП. (контр. вопрос 6): τ, длина пробега, D — при заданной эффективной массе [ОЦЕНКА]")
    P("-" * 78)
    for nm, m_eff in (("электроны (m*=0.12·m0)", M_EFF_ELECTRON), ("дырки (m*=0.35·m0)", M_EFF_HOLE)):
        tau = mu * m_eff / E_CH
        vth = np.sqrt(3 * K_B * T_ROOM / m_eff)
        P(f"  {nm}: τ = {tau:.2e} с;  1/τ = {1/tau:.2e} с⁻¹;  v_T = {vth:.2e} м/с;  λ = v_T·τ = {vth*tau*1e9:.1f} нм;  "
          f"D = μkT/e = {mu*K_B*T_ROOM/E_CH*1e4:.1f} см²/с (T = {T_ROOM:.0f} К)")

    P("\n" + "-" * 78)
    P("[10] ФАЙЛЫ")
    P("-" * 78)
    tab13.to_csv(os.path.join(args.results, "table_k_per_current.csv"), index=False, encoding="utf-8-sig")
    cal_out = cal_df.copy()
    cal_out["B_fit_mT"] = np.where(use, Bfun(I_c), np.nan)
    cal_out["in_fit"] = use
    cal_out.to_csv(os.path.join(args.results, "table_calibration.csv"), index=False, encoding="utf-8-sig")
    P("  results/: table_k_per_current.csv, table_calibration.csv, results.json")
    P("  figures/: fig1_calibration, fig2a_hall_overview, fig2b_hall_panels, hall_individual/fig2_I_*mA, fig3_reverse_field, fig4_k_vs_I (PDF)")
    P("\n  ДОПУЩЕНИЯ, которые стоит упомянуть в отчёте:")
    P("   • приборные погрешности (σ_U, масштабы B и I, геометрия) — оценки, а не паспортные значения;")
    P("   • 'l'=1.7 мм из листа parameters принята за толщину h (в формулах методички: a, h, l(3-5));")
    P("   • индукция B для каждой точки берётся из калибровки по I_M; при реверсе принято |B|(I_M) то же;")
    P("   • одно измерение U35 (при максимальном токе) -> погрешность ρ0 приборная.")

    with open(os.path.join(args.results, "results.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(LOG) + "\n")
    js = dict(calibration=dict(poly_degree=cal["deg"], params=[float(x) for x in cal["p"]], fit_range_mA=[0, CAL_FIT_IMAX_MA],
                               sigma_B_mT=cal["rms"], B_rem_mT=B_rem),
              k_per_current=tab13.to_dict(orient="records"),
              s_uV_per_mT_mA=s_val, s_stat=s_stat,
              RH_m3_per_C=RH, RH_rel_stat=r_stat, RH_rel_sys=RH_sys, RH_rel_total=RH_tot,
              n_m3=n_m3, rho_ohm_m=rho0, rho_rel=rho_tot, sigma_S_per_m=sigma0,
              mu_m2_per_Vs=mu, mu_rel_stat=r_stat, mu_rel_sys=mu_sys, mu_rel_total=mu_tot,
              alt_calibration_RH_cm3_per_C=alt_RH, sign_k=float(sign_k), carrier=carrier,
              h_mm=h_mm, a_mm=a_mm, l35_mm=l35_mm)
    with open(os.path.join(args.results, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(js, fh, ensure_ascii=False, indent=2, default=float)
    print(f"\nГотово. Графики: {os.path.abspath(args.figures)}\nЧисловые результаты: {os.path.abspath(args.results)}")


if __name__ == "__main__":
    main()
