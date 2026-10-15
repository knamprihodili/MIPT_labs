#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Лаб. 3.5.2: оцифровка зондовых ВАХ с фотографий экрана осциллографа GOS-620 (режим X-Y).

Что делает скрипт (для каждого давления):
  1. Находит линии масштабной сетки экрана (преобразование Хафа по тёмным линиям),
     нумерует их от центральных осей (центр задаётся вручную в CENTER, см. ниже)
     и строит гомографию «пиксели -> деления». Так учитывается перспектива и наклон фото.
  2. Выделяет луч: пиксели, у которых зелёный канал заметно больше синего (луч бирюзовый,
     экран и блик - синие), яркие и не бежевые (рамка).
  3. Переводит пиксели луча в деления и для каждой полоски по X шириной DX дел.
     находит: среднюю линию (взвешенное среднее по яркости), нижнюю и верхнюю границы
     (5-й и 95-й процентили). Если на экране петля (фазовый сдвиг), средняя линия
     идёт между ветвями, а ширина петли идёт в погрешность тока.

Выход:  data/vah_<P>.csv  (x_div, y_div, y_lo_div, y_hi_div, n_pix)
        figures/digitization_check_<P>.pdf - по одному файлу на давление:
        слева фото с наложенной сеткой и линией, справа оцифрованная кривая.

Нужно: numpy, matplotlib, opencv-python  (pip install numpy matplotlib opencv-python)
Запуск:  python scripts/digitize_vah.py   (из корня папки лабы)
"""
import os
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCR = os.path.join(ROOT, "data", "screens")

# давление (торр) -> центр экрана (x центральной вертикали, y центральной горизонтали)
# в пикселях исходного изображения - задан по фото вручную, остальные линии ищутся сами
CENTER = {
    "8e-2":   (653, 517),
    "9e-2":   (791, 642),
    "1e-1":   (696, 580),
    "1.5e-1": (675, 562),
    "2e-1":   (585, 502),
}
DX = 0.1      # ширина полоски по X, дел.


def dark_mask(img):
    """Тёмные линии сетки: пиксель темнее локальной медианы на 25 уровней."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bg = cv2.medianBlur(g, 31).astype(np.float32)
    return (((bg - g) > 25) * 255).astype(np.uint8)


def hough_families(mask):
    L = cv2.HoughLines(mask, 1, np.pi / 720, int(0.35 * min(mask.shape)))[:, 0, :]
    V, H = [], []
    for r, t in L:
        if t > np.pi / 2:
            r, t = -r, t - np.pi
        if abs(t) < 0.12:
            V.append((r, t))
        elif abs(t - np.pi / 2) < 0.12:
            H.append((r, t))
    return np.array(V), np.array(H)


def cluster(F, gap=25):
    F = F[np.argsort(F[:, 0])]
    out, cur = [], [F[0]]
    for f in F[1:]:
        if f[0] - cur[-1][0] < gap:
            cur.append(f)
        else:
            out.append(np.mean(cur, 0))
            cur = [f]
    out.append(np.mean(cur, 0))
    return np.array(out)


def index_lines(C, c0):
    """Номер линии = округлённое расстояние от центральной в шагах сетки."""
    c = np.argmin(abs(C[:, 0] - c0))
    sp = np.median(np.diff(C[:, 0]))
    f = (C[:, 0] - C[c, 0]) / sp
    idx = np.round(f).astype(int)
    ok = abs(f - idx) < 0.2
    return idx[ok], C[ok]


def intersect(l1, l2):
    (r1, t1), (r2, t2) = l1, l2
    A = np.array([[np.cos(t1), np.sin(t1)], [np.cos(t2), np.sin(t2)]])
    return np.linalg.solve(A, [r1, r2])


def grid_homography(img, center):
    m = dark_mask(img)
    V, H = hough_families(m)
    iv, CV = index_lines(cluster(V), center[0])
    ih, CH = index_lines(cluster(H), center[1])
    src, dst = [], []
    for a, lv in zip(iv, CV):
        for b, lh in zip(ih, CH):
            if abs(a) <= 5 and abs(b) <= 4:
                src.append(intersect(lv, lh))
                dst.append((a, -b))            # ось Y вверх
    src, dst = np.float32(src), np.float32(dst)
    Hm, _ = cv2.findHomography(src, dst, 0)
    res = np.linalg.norm(cv2.perspectiveTransform(src[None], Hm)[0] - dst, axis=1)
    return Hm, res, len(src)


def beam_pixels(img, Hm, gb_min=0, g_min=150):
    B, G, R = [img[:, :, i].astype(int) for i in range(3)]
    score = cv2.GaussianBlur((G - B).astype(np.float32), (5, 5), 0)
    ys, xs = np.nonzero((score > gb_min) & (G > g_min) & ((R < G - 40) | (G > 235)))
    P = cv2.perspectiveTransform(np.float32(np.c_[xs, ys])[None], Hm)[0]
    w = G[ys, xs].astype(float)
    ok = (abs(P[:, 0]) < 5.3) & (abs(P[:, 1]) < 4.02)
    return P[ok], w[ok], (xs[ok], ys[ok])


def midline(P, w, dx=DX):
    x, y = P[:, 0], P[:, 1]
    out = []
    for a in np.arange(np.floor(x.min() / dx) * dx, x.max(), dx):
        s = (x >= a) & (x < a + dx)
        if s.sum() < 15:
            continue
        ys, ws = y[s], w[s]
        out.append((a + dx / 2, np.average(ys, weights=ws),
                    np.percentile(ys, 5), np.percentile(ys, 95), s.sum()))
    return np.array(out)


def main():
    os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)

    for p, c in CENTER.items():
        path = os.path.join(SCR, f"p_{p}.png")
        img = cv2.imread(path)
        if img is None:
            print(f"Не удалось прочитать {path} - пропускаю")
            continue

        gb_min = 5 if p == "8e-2" else 0      # на этом фото фон чуть зеленее
        Hm, res, nint = grid_homography(img, c)
        P, w, (xs, ys) = beam_pixels(img, Hm, gb_min=gb_min)
        ml = midline(P, w)
        np.savetxt(os.path.join(ROOT, "data", f"vah_{p}.csv"), ml, delimiter=",", fmt="%.4f",
                   header="x_div,y_div,y_lo_div,y_hi_div,n_pix", comments="")
        print(f"P = {p:>7} торр: узлов сетки {nint}, макс. невязка сетки {res.max():.3f} дел., "
              f"точек линии {len(ml)}")

        # заголовок собираем заранее (обратный слэш внутри f-строки не работает до Python 3.12)
        mant, _, exp = p.partition("e")
        title = rf"$P = {mant}\cdot 10^{{{exp}}}$ торр"

        # одна строка, два столбца: фото слева, кривая справа
        fig, (a, b) = plt.subplots(1, 2, figsize=(12, 5.2),
                                   gridspec_kw={"width_ratios": [1.25, 1]})

        # контроль: фото + сетка + средняя линия
        a.imshow(img[:, :, ::-1])
        Hi = np.linalg.inv(Hm)
        for u in range(-5, 6):
            q = cv2.perspectiveTransform(np.float32([[u, -4], [u, 4]])[None], Hi)[0]
            a.plot(q[:, 0], q[:, 1], color="r" if u else "y", lw=0.8)
        for v in range(-4, 5):
            q = cv2.perspectiveTransform(np.float32([[-5, v], [5, v]])[None], Hi)[0]
            a.plot(q[:, 0], q[:, 1], color="r" if v else "y", lw=0.8)
        q = cv2.perspectiveTransform(np.float32(ml[:, :2])[None], Hi)[0]
        a.plot(q[:, 0], q[:, 1], "m-", lw=1.5)
        a.set_title(title, fontsize=20)
        a.axis("off")

        b.scatter(P[::5, 0], P[::5, 1], s=0.5, c="c")
        b.fill_between(ml[:, 0], ml[:, 2], ml[:, 3], color="0.6", alpha=0.5, lw=0,
                       label="ширина луча/петли")
        b.plot(ml[:, 0], ml[:, 1], "m-", lw=1.5, label="средняя линия")
        b.set_xlim(-5, 5); b.set_ylim(-4, 4); b.set_aspect("equal")
        b.set_xticks(range(-5, 6)); b.set_yticks(range(-4, 5)); b.grid(lw=0.4)
        b.set_xlabel("X, дел.", fontsize=20); b.set_ylabel("Y, дел.", fontsize=20)
        b.tick_params(labelsize=14)
        b.legend(loc="upper left", fontsize=12)

        plt.tight_layout()
        plt.savefig(os.path.join(ROOT, "figures", f"digitization_check_{p}.pdf"))
        plt.close(fig)


if __name__ == "__main__":
    main()