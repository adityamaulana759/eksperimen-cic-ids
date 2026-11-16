"""
Analisis tambahan untuk naskah "Cross-Dataset Generalization of Minority and
Low-Rate Attack Detection" (CIC-IDS2017 -> CSE-CIC-IDS2018).

Skrip ini TIDAK melatih ulang model. Semua angka dihitung dari folder hasil
yang dibuat notebook eksperimen_full_dataset_vscode.ipynb (mode FULL):
  <hasil>/Table_S1_per_class.csv
  <hasil>/Table_statistical_tests.csv
  <hasil>/S1/*_folds.csv
  <hasil>/S2/*_confusion.csv

Keluaran (ditulis ke <hasil>/analisis_tambahan/):
  S2_bootstrap_ci.csv        - 95% bootstrap CI macro-F1 dan MCC di S2 (Tabel 9, Subbab 3.1.4)
  sensitivity_no_infiltration.csv - macro-F1 S1/S2 tanpa Infiltration (kolom terakhir Tabel 9)
  S1_wilcoxon_holm.csv       - p Wilcoxon vs A1 + koreksi Holm (kolom terakhir Tabel 8)
  S1_corrected_t_holm.csv    - corrected resampled t-test (Nadeau-Bengio) + Holm (Subbab 3.1.5)
  S2_per_class_precision.csv - presisi/recall/F1 per kelas di S2 (Tabel 10)
  S2_benign_false_alarms.csv - persentase flow Benign yang dilabeli serangan (Pembahasan)

Pemakaian:
  python analisis_tambahan.py --hasil "path/ke/hasil_ringkas_full"
"""
import argparse, glob, os
import numpy as np
import pandas as pd
from scipy import stats

ABLASI = ['A0 Baseline', 'A1 Full model', 'A2 Without temporal', 'A3 Without volume',
          'A4 Without resampling', 'A5 Without class weighting', 'A6 Without sub-classifier',
          'A7 SMOTE', 'A7 Borderline-SMOTE', 'A7 ADASYN', 'A7 SMOTE-ENN']
REF = 'A1 Full model'


def nama_dari_file(path, akhiran):
    """'A7_Borderline_SMOTE_confusion.csv' -> 'A7 Borderline-SMOTE'."""
    s = os.path.basename(path)[: -len(akhiran)]
    peta = {'A6_Without_sub_classifier': 'A6 Without sub-classifier',
            'A7_Borderline_SMOTE': 'A7 Borderline-SMOTE', 'A7_SMOTE_ENN': 'A7 SMOTE-ENN'}
    return peta.get(s, s.replace('_', ' '))


def metrik(C, buang=None):
    """Macro-F1, macro-F1 tanpa satu kelas, dan MCC multikelas dari confusion matrix."""
    C = np.asarray(C, float)
    tp = np.diag(C); fp = C.sum(0) - tp; fn = C.sum(1) - tp
    with np.errstate(invalid='ignore', divide='ignore'):
        p = np.where(tp + fp > 0, tp / (tp + fp), 0)
        r = np.where(tp + fn > 0, tp / (tp + fn), 0)
        f = np.where(p + r > 0, 2 * p * r / (p + r), 0)
    sisa = [i for i in range(len(f)) if i != buang]
    t = C.sum(1); pr = C.sum(0); s = C.sum(); c = tp.sum()
    mcc = (c * s - (pr * t).sum()) / np.sqrt((s**2 - (pr**2).sum()) * (s**2 - (t**2).sum()))
    return f.mean(), f[sisa].mean(), mcc


def holm(pvals):
    """Koreksi Holm (step-down) untuk dict {nama: p}."""
    urut = sorted(pvals.items(), key=lambda x: x[1]); m = len(urut); hasil = {}; maks = 0.0
    for i, (k, p) in enumerate(urut):
        maks = max(maks, min(1.0, p * (m - i))); hasil[k] = maks
    return hasil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hasil', required=True, help='folder hasil_ringkas_full')
    ap.add_argument('--B', type=int, default=1000, help='jumlah resample bootstrap')
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()
    H = a.hasil; OUT = os.path.join(H, 'analisis_tambahan'); os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    # ---- 1. Bootstrap CI S2 + macro-F1 tanpa Infiltration --------------------------
    rows = []
    for fn in sorted(glob.glob(os.path.join(H, 'S2', '*_confusion.csv'))):
        nama = nama_dari_file(fn, '_confusion.csv')
        C = pd.read_csv(fn, index_col=0); inf = list(C.columns).index('Infiltration')
        f1, f1_noinf, mcc = metrik(C.values, inf)
        sel = C.values.ravel(); n = int(sel.sum())
        bs = np.array([metrik(rng.multinomial(n, sel / n).reshape(C.shape), inf) for _ in range(a.B)])
        lo, hi = np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0)
        b = C.loc['Benign']; fa = b.sum() - b['Benign']
        rows.append(dict(configuration=nama, macro_f1=f1, f1_ci_lo=lo[0], f1_ci_hi=hi[0],
                         macro_f1_no_infiltration=f1_noinf, mcc=mcc, mcc_ci_lo=lo[2], mcc_ci_hi=hi[2],
                         benign_flows=int(b.sum()), benign_false_alarms=int(fa),
                         benign_false_alarm_pct=100 * fa / b.sum()))
    s2 = pd.DataFrame(rows).set_index('configuration')
    s2[['macro_f1', 'f1_ci_lo', 'f1_ci_hi', 'mcc', 'mcc_ci_lo', 'mcc_ci_hi']].to_csv(os.path.join(OUT, 'S2_bootstrap_ci.csv'))
    s2[['benign_flows', 'benign_false_alarms', 'benign_false_alarm_pct']].to_csv(os.path.join(OUT, 'S2_benign_false_alarms.csv'))

    s1c = pd.read_csv(os.path.join(H, 'Table_S1_per_class.csv'))
    s1_noinf = s1c[s1c['class'] != 'Infiltration'].groupby('configuration').f1_mean.mean()
    sens = pd.DataFrame({'macro_f1_S1_no_infiltration': s1_noinf,
                         'macro_f1_S2_no_infiltration': s2.macro_f1_no_infiltration})
    sens['gap_no_infiltration'] = sens.iloc[:, 0] - sens.iloc[:, 1]
    sens.to_csv(os.path.join(OUT, 'sensitivity_no_infiltration.csv'))

    # ---- 2. Presisi per kelas S2 -----------------------------------------------------
    s2c = pd.read_csv(os.path.join(H, 'Table_S2_per_class.csv'))
    s2c.pivot(index='configuration', columns='class', values=['recall', 'precision', 'f1']).to_csv(
        os.path.join(OUT, 'S2_per_class_precision.csv'))

    # ---- 3. Wilcoxon + Holm (dari tabel uji notebook) ---------------------------------
    st = pd.read_csv(os.path.join(H, 'Table_statistical_tests.csv'))
    st = st[(st.reference == REF) & (st.metric == 'macro_f1') & st.configuration.isin(ABLASI)].set_index('configuration')
    hw = holm(st.wilcoxon_p.to_dict())
    pd.DataFrame({'wilcoxon_p': st.wilcoxon_p, 'wilcoxon_p_holm': pd.Series(hw)}).to_csv(
        os.path.join(OUT, 'S1_wilcoxon_holm.csv'))

    # ---- 4. Corrected resampled t-test (Nadeau & Bengio, 2003) + Holm -------------------
    folds = {}
    for fn in glob.glob(os.path.join(H, 'S1', '*_folds.csv')):
        d = pd.read_csv(fn); folds[d.configuration.iloc[0]] = d.set_index('fold').macro_f1
    ref = folds[REF]; rasio = 1 / 4  # n_uji / n_latih untuk 5-fold CV
    out = {}
    for c in ABLASI:
        if c == REF: continue
        d = (folds[c] - ref).values; n = len(d)
        t = d.mean() / np.sqrt((1 / n + rasio) * d.var(ddof=1))
        out[c] = dict(mean_diff=d.mean(), t_corrected=t, p_corrected=2 * stats.t.sf(abs(t), n - 1))
    ct = pd.DataFrame(out).T
    ct['p_corrected_holm'] = pd.Series(holm(ct.p_corrected.to_dict()))
    ct.sort_values('p_corrected').to_csv(os.path.join(OUT, 'S1_corrected_t_holm.csv'))

    # ---- Ringkasan ---------------------------------------------------------------
    pd.set_option('display.width', 200)
    print('\n== S2 macro-F1 dengan 95% CI dan tanpa Infiltration ==')
    print(s2.loc[[c for c in ABLASI if c in s2.index], ['macro_f1', 'f1_ci_lo', 'f1_ci_hi', 'macro_f1_no_infiltration', 'benign_false_alarm_pct']].round(4))
    print('\n== Wilcoxon vs A1 (Holm) ==');  print(pd.read_csv(os.path.join(OUT, 'S1_wilcoxon_holm.csv'), index_col=0).round(4))
    print('\n== Corrected resampled t-test vs A1 (Holm) =='); print(ct.sort_values('p_corrected').round(4))
    print(f'\nSemua file tersimpan di: {OUT}')


if __name__ == '__main__':
    main()
