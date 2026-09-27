#!/usr/bin/env python3
"""
Evalua GLiClass (encoder ModernBERT-base, 151M) en la MISMA tarea y split que el resto.
Devuelve score por etiqueta; P(spam) = score normalizado entre las dos etiquetas.
"""
import math, random, sys, time, warnings, argparse
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/SMSSpamCollection")
ap.add_argument("--n", type=int, default=400)
ap.add_argument("--calibrate", type=int, default=200)
ap.add_argument("--device", default="cuda")
ap.add_argument("--model", default="knowledgator/gliclass-modern-base-v2.0-init")
ap.add_argument("--label", default="GLiClass")
A = ap.parse_args()

LABELS = ["spam or scam or unsolicited marketing", "normal personal message"]


def load(path):
    rows = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if "\t" not in line:
                continue
            lab, txt = line.split("\t", 1)
            lab = lab.strip()
            if lab in ("ham", "spam"):
                rows.append((1 if lab == "spam" else 0, txt.strip()))
    random.Random(42).shuffle(rows)
    return rows


def sigmoid(z):
    return 1 / (1 + math.exp(-z)) if z > -700 else 0.0


def auroc(ys, ps):
    pos = [p for y, p in zip(ys, ps) if y == 1]
    neg = [p for y, p in zip(ys, ps) if y == 0]
    if not pos or not neg:
        return float("nan")
    s = sum(1.0 if a > b else (0.5 if a == b else 0.0) for a in pos for b in neg)
    return s / (len(pos) * len(neg))


def ece(ys, ps, bins=10):
    tot = 0.0
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        sel = [(y, p) for y, p in zip(ys, ps) if lo < p <= hi or (i == 0 and p <= 0)]
        if not sel:
            continue
        tot += len(sel) / len(ys) * abs(sum(p for _, p in sel) / len(sel)
                                        - sum(y for y, _ in sel) / len(sel))
    return tot


def f1_at(ys, ps, th):
    tp = sum(1 for y, p in zip(ys, ps) if p >= th and y == 1)
    fp = sum(1 for y, p in zip(ys, ps) if p >= th and y == 0)
    fn = sum(1 for y, p in zip(ys, ps) if p < th and y == 1)
    tn = sum(1 for y, p in zip(ys, ps) if p < th and y == 0)
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return ((2 * pr * rc / (pr + rc)) if pr + rc else 0.0,
            (tp + tn) / len(ys), pr, rc, (tp, fp, fn, tn))


def report(tag, ys, ps):
    f1, acc, pr, rc, cm = f1_at(ys, ps, 0.5)
    bf1, bth = max(((f1_at(ys, ps, t / 100)[0], t / 100) for t in range(1, 100)))
    brier = sum((p - y) ** 2 for y, p in zip(ys, ps)) / len(ys)
    print("  %-14s AUROC=%.4f  acc=%.4f P=%.4f R=%.4f F1=%.4f | bestF1=%.4f@%.2f | Brier=%.4f ECE=%.4f"
          % (tag, auroc(ys, ps), acc, pr, rc, f1, bf1, bth, brier, ece(ys, ps)))
    print("                 TP=%d FP=%d FN=%d TN=%d" % cm)


def fit_platt(ys, zs):
    a, b, lr, n = 1.0, 0.0, 0.05, len(ys)
    for _ in range(3000):
        ga = gb = 0.0
        for y, z in zip(ys, zs):
            d = sigmoid(a * z + b) - y
            ga += d * z; gb += d
        a -= lr * ga / n; b -= lr * gb / n
    return a, b


def main():
    from gliclass import GLiClassModel, ZeroShotClassificationPipeline
    from transformers import AutoTokenizer
    t0 = time.time()
    model = GLiClassModel.from_pretrained(A.model)
    tok = AutoTokenizer.from_pretrained(A.model)
    pipe = ZeroShotClassificationPipeline(model, tok, classification_type="single-label",
                                          device=A.device)
    print("\n### %s (%s)  device=%s" % (A.label, A.model.split("/")[-1], A.device))
    print("  carga modelo   %.1fs" % (time.time() - t0))

    rows = load(A.data)
    test, calib = rows[:A.n], rows[-A.calibrate:] if A.calibrate else []
    print("  test n=%d spam=%d" % (len(test), sum(y for y, _ in test)))

    def p_spam(msg):
        r = pipe(msg[:2000], LABELS, threshold=0.0)[0]
        d = {x["label"]: float(x["score"]) for x in r}
        a, b = d.get(LABELS[0], 0.0), d.get(LABELS[1], 0.0)
        if a + b <= 0:
            return None
        return min(max(a / (a + b), 1e-6), 1 - 1e-6)

    t0 = time.time()
    ys, ps = [], []
    for y, m in test:
        p = p_spam(m)
        if p is None:
            continue
        ys.append(y); ps.append(p)
    el = time.time() - t0
    print("  velocidad      %.1f msg/s (%.1fs)" % (len(ys) / el, el))
    report("sin calibrar", ys, ps)

    if calib:
        cys, czs = [], []
        for y, m in calib:
            p = p_spam(m)
            if p is None:
                continue
            cys.append(y); czs.append(math.log(p / (1 - p)))
        if cys:
            a, b = fit_platt(cys, czs)
            print("  Platt          a=%.3f b=%.3f" % (a, b))
            report("platt", ys, [sigmoid(a * math.log(p / (1 - p)) + b) for p in ps])


main()
