#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Анализ АЧХ / ФЧХ колебательного контура по записям осциллографа (CSV),
снятым при линейной развёртке частоты генератора (режим Sweep / ГКЧ).

Входные файлы:
  r1.csv, r2.csv              - один канал (CH1, выход контура) при развёртке;
                                 огибающая сигнала = АЧХ (для двух значений R).
  fchh410.csv, fchh2040.csv   - два канала (CH1 - вход, CH2 - выход) при
                                 развёртке, те же R, что и в r1/r2; разность
                                 фаз CH2-CH1 как функция частоты = ФЧХ.

Ось частоты:
  - Для r1.csv/r2.csv параметры развёртки ИЗВЕСТНЫ и заданы явно (Start/Stop/
    Time генератора, см. SWEEP_* или --start/--stop/--time): f(t) = Start +
    (Stop-Start)/Time * t. Скорость 800 Гц/с, посчитанная из этих параметров,
    совпадает со скоростью, независимо измеренной прямо по записи - это
    подтверждает, что ось верна.
  - Для fchh410.csv/fchh2040.csv параметры развёртки генератора ПОКА НЕ
    заданы (измеренная по данным скорость развёртки у них другая - около
    898 Гц/с, а не 800 Гц/с, как у r1/r2, то есть настройки были другие).
    Для них частота восстанавливается автоматически по самой записи: скорость
    развёртки - из видимой (свёрнутой) частоты, а начальная частота - из
    ближайшей к RESONANCE_HINT_HZ зоны (см. ниже), чтобы фазовая кривая легла
    в тот же диапазон частот, где на АЧХ виден резонанс. Как только появятся
    точные Start/Stop/Time для этих файлов, их нужно задать так же, как для
    r1/r2 - через SWEEP2_* / отдельные аргументы, и автоподбор станет не нужен.

Частота дискретизации осциллографа (1 кГц) ниже частоты сигнала, поэтому
записанный сигнал ПЕРЕДИСКРЕТИЗОВАН (alias). Мгновенная амплитуда и фаза всё
равно корректно восстанавливаются преобразованием Гильберта, НО ровно на
частотах, кратных fs/2 = 500 Гц, оценка вырождается - там на графиках видны
провалы АЧХ и выбросы ФЧХ (отмечены пунктирными линиями). Это сырые данные,
сглаживание/интерполяция через эти точки не делается.

Запуск:
  python analyze_rlc.py --dir ПАПКА [--start 2990 --stop 10990 --time 10] [--no-show]
"""
import argparse
import sys as _sys
if _sys.version_info < (3, 8):
    raise SystemExit('Нужен Python 3.8+ (сейчас %s)' % _sys.version.split()[0])
import os
import re
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import hilbert, medfilt, savgol_filter
from scipy.ndimage import uniform_filter1d

# ----------------------------- НАСТРОЙКИ ------------------------------------
DATA_DIR = r"C:\Utilities\Projects\MIPT_labs\lab_3.2.4+5\data"
FIGURES_DIR = r"C:\Utilities\Projects\MIPT_labs\lab_3.2.4+5\figures"
AFC_FILES = ["r1.csv", "r2.csv"]
AFC_LABELS = {"r1.csv": "410 Ом", "r2.csv": "2100 Ом"}
PHASE_FILES = ["fchh410.csv", "fchh2040.csv"]
PHASE_LABELS = {"fchh410.csv": "410 Ом", "fchh2040.csv": "2100 Ом"}

# r1/r2: параметры развёртки генератора ИЗВЕСТНЫ (АКИП-3409/4, режим ГКЧ)
SWEEP_START = 2990.0   # Гц
SWEEP_STOP = 10990.0   # Гц
SWEEP_TIME = 10.0      # с

# fchh*: параметры развёртки пока не даны -> частота восстанавливается
# автоматически; ориентир для выбора зоны (кратной 1000 Гц) - резонанс АЧХ.
RESONANCE_HINT_HZ = 6000.0

ENCODINGS = ("utf-8-sig", "cp1251", "latin-1")
DEFAULT_DT = 1e-3            # с, если в шапке нет 'Time interval'
EDGE = 40                    # отсчётов у краёв записи, отбрасываемых (краевые эффекты Гильберта)
ALIAS_MARGIN = 30            # Гц: около 0 и fs/2 фаза/амплитуда по Гильберту ненадёжны
MODEL_TOL = 40                # Гц: допуск расхождения видимой частоты с моделью развёртки
HALF_MASK = 50                # Гц: полоса вокруг частот, кратных fs/2 (500 Гц), отбрасывается
F_GEN_REL = 1e-4             # АКИП-3409/4: погрешность установки частоты 1e-4 (паспорт)
SCOPE_GAIN_REL = 0.03        # ADS-6124H: отн. погрешность вертикального канала - ОЦЕНКА, уточнить по паспорту (--scope-err)
SIG_WIN = 101                # отсчётов: окно оценки разброса
PHASE_WIN = 31                # отсчётов усреднения фазы (оценка мгновенной разности фаз)
# ----------------------------------------------------------------------------


# ------------------------------ ПАРСИНГ CSV ---------------------------------
def _read_lines(path):
    for enc in ENCODINGS:
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read().splitlines()
        except UnicodeDecodeError:
            continue
    raise IOError(f"Не удалось прочитать {path}")


def _parse_row(line, delim):
    line = line.strip()
    if not line:
        return None
    toks = [t.strip() for t in line.split(delim)]
    if delim != ",":
        toks = [t.replace(",", ".") for t in toks]
    while toks and toks[-1] == "":
        toks.pop()
    if not toks:
        return None
    try:
        return [float(t) for t in toks]
    except ValueError:
        return None


def _detect_delimiter(lines):
    best, best_score = ",", -1
    for d in (";", "\t", ","):
        score = sum(1 for ln in lines[:500]
                    if (r := _parse_row(ln, d)) is not None and len(r) >= 2)
        if score > best_score:
            best, best_score = d, score
    return best


def _numeric_block(lines, delim, min_cols):
    best, cur = [], []
    for ln in lines:
        row = _parse_row(ln, delim)
        if row is None or len(row) < min_cols or (cur and len(row) != len(cur[0])):
            if len(cur) > len(best):
                best = cur
            cur = []
            if row is None or len(row) < min_cols:
                continue
        cur.append(row)
    if len(cur) > len(best):
        best = cur
    return np.array(best, dtype=float)


_UNIT = {"ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1.0}


def _meta_dt(lines):
    for ln in lines[:40]:
        if ln.lower().startswith("time interval"):
            m = re.search(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*([nmu]?s)\b", ln, re.I)
            if m:
                return float(m.group(1)) * _UNIT[m.group(2).lower()]
    return DEFAULT_DT


def _column_scales(lines, ncols):
    for ln in lines[:60]:
        if re.search(r"index", ln, re.I) and "(" in ln:
            heads = [h.strip() for h in re.split(r"[;,\t]", ln)]
            sc = []
            for h in heads:
                if re.search(r"\(mV\)", h, re.I):
                    sc.append(1e-3)
                elif re.search(r"\(V\)", h, re.I):
                    sc.append(1.0)
                else:
                    sc.append(None)
            if len(sc) == ncols:
                return sc
    return [None] * ncols


def load_scope(path):
    """Возвращает (dt, [ch1, ch2, ...]) в вольтах."""
    lines = _read_lines(path)
    delim = _detect_delimiter(lines)
    data = _numeric_block(lines, delim, 2)
    if data.size == 0:
        raise ValueError(f"{path}: числовые данные не найдены")
    sc = _column_scales(lines, data.shape[1])
    has_index = data.shape[1] >= 2 and sc[0] is None and np.allclose(np.diff(data[:, 0]), 1)
    cols = range(1, data.shape[1]) if has_index else range(data.shape[1])
    chans = [data[:, c] * (sc[c] if sc[c] is not None else 1e-3) for c in cols]
    return _meta_dt(lines), chans


# --------------------------- РАЗВЁРТКА / АЛИАСИНГ ---------------------------
def alias_freq(x, fs):
    """Видимая (свёрнутая в 0..fs/2) мгновенная частота, Гц, + аналитический сигнал."""
    a = hilbert(x - np.mean(x))
    z = a[1:] * np.conj(a[:-1])
    fa = np.abs(np.angle(z)) * fs / (2 * np.pi)
    fa = np.append(fa, fa[-1])
    return medfilt(fa, 15), a


def fold(f, fs):
    """Истинная частота -> видимая (0..fs/2)."""
    g = np.mod(f, fs)
    return np.minimum(g, fs - g)


def reconstruct_frequency(fa, fs, f0_hint):
    """
    Автоподбор для файлов без известных параметров развёртки: линейная
    монотонно растущая развёртка f(t) = f0 + r*t. Скорость r измеряется по
    расстоянию между пересечениями уровня fs/4 видимой (свёрнутой) частотой
    fa - это не гадание, а прямое измерение по данным. Начальная частота f0
    определяется с точностью до кратного 1000 Гц; конкретная зона выбирается
    ближайшей к f0_hint.
    """
    n = len(fa)
    lvl = fs / 4
    fm = medfilt(fa, 41)
    state = np.where(fm < lvl - 0.2 * fs, -1, np.where(fm > lvl + 0.2 * fs, 1, 0))
    tc_list, rising_list, last, last_i = [], [], 0, None
    for i, st in enumerate(state):
        if st == 0:
            continue
        if last != 0 and st != last:
            seg = fm[last_i:i + 1]
            j = last_i + int(np.argmin(np.abs(seg - lvl)))
            tc_list.append(float(j))
            rising_list.append(1 if st > 0 else 0)
        last, last_i = st, i
    tc = np.array(tc_list)
    rising = np.array(rising_list)
    ok = (tc > EDGE) & (tc < n - EDGE)
    tc, rising = tc[ok], rising[ok]
    if len(tc) < 3:
        raise ValueError("слишком мало периодов пилообразной видимой частоты для оценки развёртки")

    step = np.median(np.diff(tc))
    k = np.round((tc - tc[0]) / step).astype(int)
    keep = np.ones(len(tc), bool)
    for _ in range(5):
        p = np.polyfit(k[keep], tc[keep], 1)
        res = tc - np.polyval(p, k)
        keep = np.abs(res) < 0.05 * abs(p[0]) + 5
        if keep.sum() < 3:
            raise ValueError("не удалось согласовать пересечения с линейной развёрткой")
    step, t0 = p[0], p[1]
    rate_per_sample = 500.0 / step

    j0 = int(np.argmax(keep))
    base = 250.0 if rising[j0] else 750.0
    t_first = np.polyval(p, k[j0])
    f_at0_offset = -rate_per_sample * t_first
    m = np.round((f0_hint - f_at0_offset - base) / 1000.0)
    f_first = base + 1000.0 * m
    f = f_first + rate_per_sample * (np.arange(n) - t_first)
    return f, rate_per_sample * fs


def sweep_from_params(n, dt, start, stop, time):
    rate = (stop - start) / time
    t = np.arange(n) * dt
    return start + rate * t, rate


def calibrate_sync_offset(f, fa, fs, delay_lo, delay_hi, dt, step_s=0.002):
    """
    Известные Start/Stop/Time задают частоту с точностью до момента, когда
    относительно начала записи фактически стартовала развёртка (задержка
    запуска/синхронизации между генератором и осциллографом). Эта задержка
    ищется прямым сравнением с измеренной по данным видимой частотой fa -
    небольшой скользящий поиск (не переподбор зоны, а именно синхронизация),
    ограниченный max_delay_s.
    """
    n = len(f)
    best_delay, best_err = 0.0, np.inf
    delays = np.arange(delay_lo, delay_hi + 1e-9, step_s)
    fs_half = fs / 2
    for delay in delays:
        shift = int(round(delay / dt))
        if shift >= 0:
            a, b = fa[shift:], fold(f, fs)[: n - shift] if shift else fold(f, fs)
        else:
            a, b = fa[: n + shift], fold(f, fs)[-shift:]
        if len(a) < 200:
            continue
        d = np.abs(a - b)
        d = np.minimum(d, fs_half - d) if False else d  # fold() already в [0, fs/2]
        err = np.median(d)
        if err < best_err:
            best_err, best_delay = err, delay
    return best_delay, best_err


def sweep_common(x1, dt, n, sweep_params=None, f0_hint=None):
    """
    Возвращает fs, f (истинная частота по отсчётам), rate, аналитический
    сигнал CH1, маску valid (доверять амплитуде/фазе) и маску нечётной зоны
    Найквиста (знак фазы там зеркалится).
    Если sweep_params=(start,stop,time) заданы - частота берётся напрямую из
    них (r1/r2). Иначе - автоматически реконструируется по данным (fchh*).
    """
    fs = 1.0 / dt
    fa, a1 = alias_freq(x1, fs)
    if sweep_params is not None:
        f, rate = sweep_from_params(n, dt, *sweep_params)
        # запись не может начаться РАНЬШЕ старта развёртки -> f(0) >= Start -> delay <= 0.
        # Видимая частота определяет Start лишь по модулю 1000 Гц (период 1000/rate с),
        # поэтому ищем ровно в одном периоде [-1000/rate, 0] - однозначно, без зеркальной копии
        # со сдвигом на 1000 Гц (так r2 раньше уезжал влево на 1000 Гц).
        period = 1000.0 / rate
        delay, err = calibrate_sync_offset(f, fa, fs, -period, 0.0, dt=dt)
        if abs(delay) > 1e-9:
            f = f - rate * delay  # знак: см. calibrate_sync_offset
    else:
        f, rate = reconstruct_frequency(fa, fs, f0_hint)

    mism = np.abs(fa - fold(f, fs))
    valid = (mism < MODEL_TOL) & (fa > ALIAS_MARGIN) & (fa < fs / 2 - ALIAS_MARGIN)
    valid &= np.abs(f - (fs / 2) * np.round(f / (fs / 2))) > HALF_MASK
    valid[:EDGE] = False
    valid[-EDGE:] = False
    inrange = np.ones(n, bool)
    inrange[:EDGE] = False
    inrange[-EDGE:] = False

    # конец записи короче времени развёртки - резкое падение амплитуды CH1
    # отмечает точку, дальше которой частотная модель недостоверна
    env = medfilt(np.abs(a1), 101)
    w = 80
    ratio = env[w:] / np.maximum(env[:-w], 1e-12)
    cut = np.where(ratio[100:] < 0.5)[0]
    if len(cut):
        valid[100 + int(cut[0]):] = False
        inrange[100 + int(cut[0]):] = False

    inverted = (np.floor(f / (fs / 2)) % 2) == 1
    return fs, f, rate, a1, valid, inverted, inrange


def rolling_std(r, win=SIG_WIN):
    """Скользящее СКО остатка r; NaN в r игнорируются (считаются по доступным точкам)."""
    ok = np.isfinite(r)
    x = np.where(ok, r, 0.0)
    w = ok.astype(float)
    cnt = np.maximum(uniform_filter1d(w, win), 1e-9)
    m = uniform_filter1d(x, win) / cnt
    v = uniform_filter1d(x ** 2, win) / cnt - m ** 2
    sig = np.sqrt(np.maximum(v, 0.0))
    return np.where(cnt > 0.3, sig, np.nan)


def wrap180(d):
    return (d + 180.0) % 360.0 - 180.0


# ------------------------------ АНАЛИЗ --------------------------------------
def analyze_afc(path, sweep_params):
    dt, ch = load_scope(path)
    n = len(ch[0])
    fs, f, rate, a1, valid, _, inrange = sweep_common(ch[0], dt, n, sweep_params=sweep_params)
    env_raw = uniform_filter1d(medfilt(np.abs(a1), 101), 21)   # огибающая по Гильберту, без доп. сглаживания
    U = np.where(valid, env_raw, np.nan)
    # статистический разброс единичных отсчётов огибающей около её сглаженного значения
    sig_stat = rolling_std(np.where(valid, np.abs(a1) - env_raw, np.nan))
    return dict(f=f[inrange], U=U[inrange], sig_stat=sig_stat[inrange], rate=rate, f0=f[0], f1=f[-1])


def _phase_pass(x1, x2, dt, n, f0_hint):
    fs, f, rate, a1, valid, inv, inrange = sweep_common(x1, dt, n, f0_hint=f0_hint)
    a2 = hilbert(x2 - np.mean(x2))
    C = a2 * np.conj(a1)
    Cs = uniform_filter1d(C.real, PHASE_WIN) + 1j * uniform_filter1d(C.imag, PHASE_WIN)
    dphi = np.degrees(np.angle(Cs))
    dphi = np.where(inv, -dphi, dphi)
    dphi_inst = np.degrees(np.angle(C))                 # мгновенная (без усреднения) разность фаз
    dphi_inst = np.where(inv, -dphi_inst, dphi_inst)
    sig_phi = rolling_std(wrap180(dphi_inst - dphi))    # разброс единичных отсчётов около усреднённой ФЧХ
    K = uniform_filter1d(medfilt(np.abs(a2), 51), 21) / np.maximum(
        uniform_filter1d(medfilt(np.abs(a1), 51), 21), 1e-12)
    valid = valid & (np.abs(a1) > 0.05 * np.nanmedian(np.abs(a1)))
    return f, rate, valid, dphi, K, inrange, sig_phi


def _steepest_crossing(f, dphi, valid):
    """
    Частота наибольшей крутизны фазовой кривой (переход через резонанс).
    Кривая передискретизируется на равномерную сетку по частоте и заметно
    сглаживается перед взятием производной, чтобы шум на пологих участках
    (например, у краёв записи) не давал ложный "самый крутой" пик - реальный
    переход в контуре растянут на сотни герц и всегда заметно круче шума.
    """
    ok = valid & np.isfinite(dphi)
    if ok.sum() < 50:
        return None
    fo, po = f[ok], dphi[ok]
    order = np.argsort(fo)
    fo, po = fo[order], po[order]
    grid = np.linspace(fo[0], fo[-1], 400)
    pg = np.interp(grid, fo, po)
    win = max(11, (len(grid) // 15) | 1)
    pg = savgol_filter(pg, win, 2)
    d = np.abs(np.gradient(pg, grid))
    # не искать переход у самых краёв восстановленного диапазона (мало данных, шумно)
    lo, hi = int(0.08 * len(grid)), int(0.92 * len(grid))
    if hi <= lo:
        return None
    i = lo + int(np.argmax(d[lo:hi]))
    return grid[i]


def analyze_phase(path, resonance_hint):
    """
    Развёртка для fchh* не задана явно, поэтому начальная частота (с точностью
    до кратного 1000 Гц) подбирается так, чтобы точка наибольшей крутизны ФЧХ
    (переход через резонанс) совпала с resonance_hint - независимым ориентиром
    от АЧХ. Скорость развёртки при этом измеряется по данным, а не гадается.
    """
    dt, ch = load_scope(path)
    if len(ch) < 2:
        raise ValueError(f"{path}: нужны два канала")
    x1, x2 = ch[0], ch[1]
    n = len(x1)

    best = None
    for shift in range(-8, 9):
        hint = resonance_hint + shift * 1000.0
        try:
            f, rate, valid, dphi, K, inrange, sig_phi = _phase_pass(x1, x2, dt, n, hint)
        except ValueError:
            continue
        fc = _steepest_crossing(f, dphi, valid)
        if fc is None:
            continue
        err = abs(fc - resonance_hint)
        if best is None or err < best[0]:
            best = (err, f, rate, valid, dphi, K, inrange, sig_phi)
    if best is None:
        raise ValueError(f"{path}: не удалось восстановить частоту развёртки")
    _, f, rate, valid, dphi, K, inrange, sig_phi = best

    clip = float(np.mean(np.abs(x2) >= 0.98 * np.max(np.abs(x2))))
    return dict(f=f[inrange], K=np.where(valid, K, np.nan)[inrange],
                dphi=np.where(valid, dphi, np.nan)[inrange],
                sig_phi=np.where(valid, sig_phi, np.nan)[inrange],
                rate=rate, f0=f[0], f1=f[-1], clip=clip)


def resonance_info(f, U):
    ok = np.isfinite(U)
    f, U = f[ok], U[ok]
    if len(f) == 0:
        return np.nan, np.nan, np.nan, np.nan
    i = int(np.argmax(U))
    fr, Um = f[i], U[i]
    lvl = Um / np.sqrt(2)
    lo = np.where(U[:i] < lvl)[0]
    hi = np.where(U[i:] < lvl)[0]
    if len(lo) and len(hi):
        f1, f2 = f[lo[-1]], f[i + hi[0]]
        return fr, Um, f2 - f1, fr / (f2 - f1)
    return fr, Um, np.nan, np.nan


# --------------------------------- MAIN -------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DATA_DIR)
    ap.add_argument("--out", default=os.path.join(FIGURES_DIR, "afc_pfc.pdf"))
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--start", type=float, default=SWEEP_START, help="r1/r2: начальная частота развёртки, Гц")
    ap.add_argument("--stop", type=float, default=SWEEP_STOP, help="r1/r2: конечная частота развёртки, Гц")
    ap.add_argument("--time", type=float, default=SWEEP_TIME, help="r1/r2: время развёртки, с")
    ap.add_argument("--resonance-hint", type=float, default=RESONANCE_HINT_HZ,
                    help="ориентир (Гц) для выбора зоны частот у fchh*, пока их развёртка не задана явно")
    ap.add_argument("--scope-err", type=float, default=SCOPE_GAIN_REL,
                    help="отн. погрешность вертикального канала ADS-6124H (по умолчанию 0.03 - оценка!)")
    ap.add_argument("--nbars", type=int, default=14, help="число точек с усами погрешностей")
    ap.add_argument("--no-errors", action="store_true", help="не рисовать погрешности")
    args = ap.parse_args()
    if args.no_show:
        matplotlib.use("Agg")
    sweep = (args.start, args.stop, args.time)
    print(f"r1/r2: развёртка {args.start:.0f} -> {args.stop:.0f} Гц за {args.time:.1f} с "
          f"({(args.stop - args.start) / args.time:.1f} Гц/с) - задано явно\n")

    plt.rcParams.update({"font.size": 14, "axes.titlesize": 20, "axes.labelsize": 20,
                     "xtick.labelsize": 20, "ytick.labelsize": 20, "legend.fontsize": 20})
    
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9))

    # ---- АЧХ ----
    print("=== АЧХ ===")
    for name in AFC_FILES:
        path = os.path.join(args.dir, name)
        try:
            r = analyze_afc(path, sweep)
        except (OSError, ValueError) as e:
            print(f"[!] {name}: {e}", file=sys.stderr)
            continue
        line, = ax1.plot(r["f"], r["U"], lw=1.2, label=AFC_LABELS.get(name, name))
        sigU = np.sqrt(r["sig_stat"] ** 2 + (args.scope_err * r["U"]) ** 2)   # полная погрешность (1σ)
        if not args.no_errors:
            ax1.fill_between(r["f"], r["U"] - 2 * sigU, r["U"] + 2 * sigU,
                             color=line.get_color(), alpha=0.22, lw=0)
            okb = np.where(np.isfinite(sigU))[0]
            jb = okb[np.linspace(0, len(okb) - 1, args.nbars + 2).astype(int)[1:-1]]
            ax1.errorbar(r["f"][jb], r["U"][jb], yerr=2 * sigU[jb], fmt="o", ms=4,
                         color=line.get_color(), mec="k", mew=0.6, ecolor="k",
                         elinewidth=1.4, capsize=4, capthick=1.4, zorder=5)
        fr, Um, bw, Q = resonance_info(r["f"], r["U"])
        print(f"{name}: развёртка {r['f0']:.0f}->{r['f1']:.0f} Гц ({r['rate']:.1f} Гц/с); "
              f"резонанс f0={fr:.0f} Гц, Umax={Um:.3f} В, Δf(-3дБ)={bw:.0f} Гц, Q≈{Q:.1f}")
        jpk = int(np.nanargmax(r["U"]))
        print(f"   Umax = {Um:.3f} ± {2 * sigU[jpk]:.3f} В (2σ: разброс отсчётов + {100*args.scope_err:.1f}% осциллограф); "
              f"f: ± {F_GEN_REL * fr:.1f} Гц (генератор, {F_GEN_REL:.0e}) ± ~1.6 Гц (синхронизация)")
    ax1.set_xlabel("f, Гц")
    ax1.set_ylabel("U, В")
    ax1.set_title("АЧХ для R1 и R2")
    ax1.grid(True, which="both", ls="--", alpha=0.6)

    # ---- ФЧХ ----
    print(f"\nfchh*: параметры развёртки не заданы явно - частота восстановлена "
          f"автоматически по данным (ориентир зоны: {args.resonance_hint:.0f} Гц)")
    print("=== ФЧХ (CH2 относительно CH1) ===")
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, name in enumerate(PHASE_FILES):
        path = os.path.join(args.dir, name)
        try:
            r = analyze_phase(path, args.resonance_hint)
        except (OSError, ValueError) as e:
            print(f"[!] {name}: {e}", file=sys.stderr)
            continue
        r["dphi"] = np.radians(r["dphi"])          # фаза выводится в радианах
        r["sig_phi"] = np.radians(r["sig_phi"])
        col = colors[i % 10]
        ax2.plot(r["f"], r["dphi"], ".", ms=2, color=col, label=PHASE_LABELS.get(name, name))
        if not args.no_errors:
            ax2.fill_between(r["f"], r["dphi"] - 2 * r["sig_phi"], r["dphi"] + 2 * r["sig_phi"],
                             color=col, alpha=0.22, lw=0)
            okb = np.where(np.isfinite(r["dphi"]) & np.isfinite(r["sig_phi"]))[0]
            jb = okb[np.linspace(0, len(okb) - 1, args.nbars + 2).astype(int)[1:-1]]
            ax2.errorbar(r["f"][jb], r["dphi"][jb], yerr=2 * r["sig_phi"][jb], fmt="s", ms=4,
                         color=col, mec="k", mew=0.6, ecolor="k", elinewidth=1.4,
                         capsize=4, capthick=1.4, zorder=5)
        ok = np.where(np.isfinite(r["dphi"]))[0]
        print(f"\n{name}: развёртка {r['f0']:.0f}->{r['f1']:.0f} Гц ({r['rate']:.1f} Гц/с)")
        if r["clip"] > 0.01:
            print(f"   [!] CH2 ограничен по амплитуде ({100*r['clip']:.0f}% отсчётов на пределе): "
                  f"K=Ch2/Ch1 занижен/недостоверен, фаза основной гармоники - нормально")
        if len(ok):
            print(f"   {'f, Гц':>10}{'K=Ch2/Ch1':>12}{'Δφ, рад':>10}{'±2σ, рад':>10}")
            for j in np.linspace(ok[0], ok[-1], 9).astype(int):
                j = ok[np.argmin(np.abs(ok - j))]
                print(f"   {r['f'][j]:>10.1f}{r['K'][j]:>12.3f}{r['dphi'][j]:>10.3f}{2*r['sig_phi'][j]:>10.3f}")
            jm = ok[len(ok) // 2]
            # (подписи файлов у кривых убраны: они дублируют легенду)

    # ---- артефакты дискретизации: частоты, кратные fs/2 = 500 Гц ----
    xmax = max(ax.dataLim.x1 for ax in (ax1, ax2) if np.isfinite(ax.dataLim.x1))
    for ax in (ax1, ax2):
        ax.set_xlim(0, xmax * 1.02)
        for k in range(1, int(xmax // 500) + 1):
            ax.axvline(500 * k, color="gray", lw=0.6, ls=":", alpha=0.7, zorder=0)
    
    ax1.legend()
    ax2.set_xlabel("f, Гц")
    ax2.set_ylabel("Δφ, рад")
    ax2.set_ylim(-1.06 * np.pi, 0.25 * np.pi)
    ax2.set_yticks(np.pi * np.array([-1, -0.75, -0.5, -0.25, 0, 0.25]))
    ax2.set_yticklabels(["−π", "−3π/4", "−π/2", "−π/4", "0", "π/4"])
    ax2.set_title("ФЧХ для R1 и R2")
    ax2.grid(True, which="both", ls="--", alpha=0.6)
    ax2.legend(markerscale=6)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"\nГрафик сохранён: {args.out}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
