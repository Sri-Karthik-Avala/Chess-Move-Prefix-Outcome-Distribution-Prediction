"""made by - Karthik"""
import os
import glob
import shutil
import warnings
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)

PIECE_LETTERS = "NBRQK"
N_SPLITS = 5
CV_SEEDS = [42, 7]
GROUP_PLY_DEPTH = 4


def extract_ngrams(tokens, ngram_max):
    grams = []
    n_tok = len(tokens)
    for n in range(1, ngram_max + 1):
        for i in range(n_tok - n + 1):
            grams.append(" ".join(tokens[i:i + n]))
    return grams


class NgramTfidf:
    def __init__(self, ngram_max=4, min_df=2, max_features=6000):
        self.ngram_max = ngram_max
        self.min_df = min_df
        self.max_features = max_features
        self.vocab_ = {}
        self.idf_ = None

    def fit(self, texts):
        doc_freq = Counter()
        total_freq = Counter()
        for t in texts:
            grams = extract_ngrams(t.split(), self.ngram_max)
            for g in set(grams):
                doc_freq[g] += 1
            for g in grams:
                total_freq[g] += 1
        terms = [g for g in doc_freq if doc_freq[g] >= self.min_df]
        if self.max_features is not None and len(terms) > self.max_features:
            terms = sorted(terms, key=lambda g: (-total_freq[g], g))[:self.max_features]
        terms = sorted(terms)
        self.vocab_ = {g: i for i, g in enumerate(terms)}
        n_docs = len(texts)
        self.idf_ = np.zeros(len(terms))
        for g, i in self.vocab_.items():
            self.idf_[i] = np.log((1 + n_docs) / (1 + doc_freq[g])) + 1.0
        return self

    def transform(self, texts):
        rows = len(texts)
        cols = len(self.vocab_)
        X = np.zeros((rows, cols), dtype=np.float64)
        for r, t in enumerate(texts):
            grams = extract_ngrams(t.split(), self.ngram_max)
            counts = Counter(grams)
            for g, c in counts.items():
                idx = self.vocab_.get(g)
                if idx is not None:
                    X[r, idx] = c
        X *= self.idf_[None, :]
        norms = np.sqrt((X ** 2).sum(axis=1, keepdims=True))
        norms[norms == 0] = 1.0
        return X / norms

    def fit_transform(self, texts):
        self.fit(texts)
        return self.transform(texts)


def locate(name):
    for base in [".", "dataset/public", "dataset/public/train", "dataset/public/test",
                 "dataset", "public", "..", "../..", "../dataset/public",
                 os.path.dirname(os.path.abspath(__file__))]:
        p = os.path.join(base, name)
        if os.path.exists(p):
            return p
    hits = glob.glob(os.path.join("**", name), recursive=True)
    return hits[0] if hits else name


def load_data():
    train = pd.read_csv(locate("train.csv"))
    test = pd.read_csv(locate("test.csv"))
    return train, test


def parse_prefix_features(move_prefix):
    toks = move_prefix.split()
    n = len(toks)
    caps = [0, 0]
    checks = [0, 0]
    castle_k = [0, 0]
    castle_q = [0, 0]
    castle_ply = [-1, -1]
    piece_counts = {p: [0, 0] for p in PIECE_LETTERS}
    pawn_moves = [0, 0]
    promo = [0, 0]
    for i, t in enumerate(toks):
        c = i % 2
        if "x" in t:
            caps[c] += 1
        if "+" in t or "#" in t:
            checks[c] += 1
        if t.startswith("O-O-O"):
            castle_q[c] += 1
            if castle_ply[c] == -1:
                castle_ply[c] = i
        elif t.startswith("O-O"):
            castle_k[c] += 1
            if castle_ply[c] == -1:
                castle_ply[c] = i
        elif t[0] in PIECE_LETTERS:
            piece_counts[t[0]][c] += 1
        else:
            pawn_moves[c] += 1
        if "=" in t:
            promo[c] += 1
    feat = {
        "n_plies": n,
        "caps_w": caps[0], "caps_b": caps[1],
        "checks_w": checks[0], "checks_b": checks[1],
        "castleK_w": castle_k[0], "castleK_b": castle_k[1],
        "castleQ_w": castle_q[0], "castleQ_b": castle_q[1],
        "castle_ply_w": castle_ply[0], "castle_ply_b": castle_ply[1],
        "pawn_w": pawn_moves[0], "pawn_b": pawn_moves[1],
        "promo_w": promo[0], "promo_b": promo[1],
    }
    for p in PIECE_LETTERS:
        feat[f"{p}_w"] = piece_counts[p][0]
        feat[f"{p}_b"] = piece_counts[p][1]
    return feat


def build_structural_df(texts):
    return pd.DataFrame([parse_prefix_features(t) for t in texts])


def build_opening_groups(texts, depth=GROUP_PLY_DEPTH):
    return np.array([" ".join(t.split()[:depth]) for t in texts])


def naive_splits(n_rows, n_splits=N_SPLITS, seed=SEED):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(kf.split(np.zeros(n_rows)))


def grouped_splits(n_rows, groups, n_splits=N_SPLITS, seed=SEED):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_rows)
    gkf = GroupKFold(n_splits=n_splits)
    splits = []
    for tr_idx, va_idx in gkf.split(np.zeros(n_rows), groups=groups[perm]):
        splits.append((perm[tr_idx], perm[va_idx]))
    return splits


def brier_score(y_true, y_pred):
    return np.mean(np.sum((y_true - y_pred) ** 2, axis=1))


def normalize_probs(p):
    p = np.clip(p, 0, None)
    s = p.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return p / s


def apply_prob_floor(preds, floor):
    if floor <= 0:
        return preds
    return normalize_probs(np.clip(preds, floor, None))


def fit_predict_ridge(alpha, Xtr, ytr, wtr, Xva):
    alphas = alpha if isinstance(alpha, (list, tuple, np.ndarray)) else [alpha] * ytr.shape[1]
    preds = np.zeros((Xva.shape[0], ytr.shape[1]))
    for k in range(ytr.shape[1]):
        m = Ridge(alpha=alphas[k])
        m.fit(Xtr, ytr[:, k], sample_weight=wtr)
        preds[:, k] = m.predict(Xva)
    return preds


def fit_predict_lgb(params, Xtr, ytr, wtr, Xva):
    preds = np.zeros((Xva.shape[0], ytr.shape[1]))
    for k in range(ytr.shape[1]):
        m = lgb.LGBMRegressor(**params, verbosity=-1, random_state=SEED)
        m.fit(Xtr, ytr[:, k], sample_weight=wtr)
        preds[:, k] = m.predict(Xva)
    return preds


def fit_predict_xgb(params, Xtr, ytr, wtr, Xva):
    preds = np.zeros((Xva.shape[0], ytr.shape[1]))
    for k in range(ytr.shape[1]):
        m = xgb.XGBRegressor(**params, verbosity=0, random_state=SEED)
        m.fit(Xtr, ytr[:, k], sample_weight=wtr)
        preds[:, k] = m.predict(Xva)
    return preds


def build_feature_block(texts_tr, texts_va, struct_tr, struct_va, cfg):
    tfidf = NgramTfidf(ngram_max=cfg["ngram_max"], min_df=cfg["min_df"], max_features=cfg["max_features"])
    Ttr = tfidf.fit_transform(texts_tr)
    Tva = tfidf.transform(texts_va)
    if cfg["use_struct"]:
        scaler = StandardScaler()
        Str_tr = scaler.fit_transform(struct_tr.values)
        Str_va = scaler.transform(struct_va.values)
        Xtr = np.hstack([Str_tr, Ttr])
        Xva = np.hstack([Str_va, Tva])
    else:
        Xtr, Xva = Ttr, Tva
    return Xtr, Xva


def oof_for_config_and_model(texts, struct_df, y, w, cfg, fit_fn, params, splits):
    oof = np.zeros_like(y)
    for tr_idx, va_idx in splits:
        Xtr, Xva = build_feature_block(texts[tr_idx], texts[va_idx],
                                        struct_df.iloc[tr_idx], struct_df.iloc[va_idx], cfg)
        oof[va_idx] = normalize_probs(fit_fn(params, Xtr, y[tr_idx], w[tr_idx], Xva))
    return oof


def oof_multi_seed(texts, struct_df, y, w, cfg, fit_fn, params, n_rows, seeds=CV_SEEDS):
    all_oof = []
    for seed in seeds:
        splits = naive_splits(n_rows, n_splits=N_SPLITS, seed=seed)
        all_oof.append(oof_for_config_and_model(texts, struct_df, y, w, cfg, fit_fn, params, splits))
    return np.mean(all_oof, axis=0)


def ridge_fit_fn(alpha, Xtr, ytr, wtr, Xva):
    return fit_predict_ridge(alpha, Xtr, ytr, wtr, Xva)


def search_blend_weights(oof_list, y, step=0.05):
    n_models = len(oof_list)
    best = (tuple(1.0 if i == 0 else 0.0 for i in range(n_models)), brier_score(y, oof_list[0]))
    grid = np.arange(0.0, 1.0 + 1e-9, step)
    if n_models == 1:
        return best
    if n_models == 2:
        for a in grid:
            b = 1.0 - a
            blend = normalize_probs(a * oof_list[0] + b * oof_list[1])
            sc = brier_score(y, blend)
            if sc < best[1]:
                best = ((a, b), sc)
        return best
    for a in grid:
        for b in grid[grid <= 1.0 - a + 1e-9]:
            c = 1.0 - a - b
            if c < -1e-9:
                continue
            c = max(c, 0.0)
            blend = normalize_probs(a * oof_list[0] + b * oof_list[1] + c * oof_list[2])
            sc = brier_score(y, blend)
            if sc < best[1]:
                best = ((a, b, c), sc)
    return best


def archive_existing_submission(out_dir, out_name="submission.csv"):
    out_path = os.path.join(out_dir, out_name)
    if not os.path.exists(out_path):
        return
    v = 1
    while os.path.exists(os.path.join(out_dir, f"submission_v{v}.csv")):
        v += 1
    shutil.move(out_path, os.path.join(out_dir, f"submission_v{v}.csv"))


def main():
    train, test = load_data()
    y = train[["white_win_rate", "draw_rate", "black_win_rate"]].values.astype(float)
    w = train["cohort_game_count"].values.astype(float)
    texts_train = train["move_prefix"].values
    texts_test = test["move_prefix"].values
    n_rows = len(train)

    struct_train = build_structural_df(texts_train)
    struct_test = build_structural_df(texts_test)

    regime_a_cfg = dict(ngram_max=4, min_df=2, max_features=6000, use_struct=True)
    regime_b_cfg = dict(ngram_max=4, min_df=2, max_features=8000, use_struct=False)
    alphas_a = [1, 2, 5, 10, 15, 20, 25, 30, 40, 60]
    alphas_b = [10, 20, 30, 50, 75, 100, 150]

    best_alpha_a, best_score_a = None, np.inf
    for alpha in alphas_a:
        oof = oof_multi_seed(texts_train, struct_train, y, w, regime_a_cfg, ridge_fit_fn, alpha, n_rows)
        sc = brier_score(y, oof)
        if sc < best_score_a:
            best_score_a, best_alpha_a = sc, alpha
    print("regime A (structural+ngram) chosen alpha:", best_alpha_a, "naive-cv brier:", best_score_a)

    draw_alphas = [1, 2, 5, 10, 15, 20, 25, 30, 40, 60]
    best_draw_alpha, best_draw_score = best_alpha_a, np.inf
    for da in draw_alphas:
        alpha_vec = [best_alpha_a, da, best_alpha_a]
        oof = oof_multi_seed(texts_train, struct_train, y, w, regime_a_cfg, ridge_fit_fn, alpha_vec, n_rows)
        sc = brier_score(y, oof)
        if sc < best_draw_score:
            best_draw_score, best_draw_alpha = sc, da
    alpha_vec_a = [best_alpha_a, best_draw_alpha, best_alpha_a]
    print("regime A draw-specific alpha:", best_draw_alpha, "(vs win/loss alpha", best_alpha_a, ") naive-cv brier:", best_draw_score)

    best_alpha_b, best_score_b = None, np.inf
    for alpha in alphas_b:
        oof = oof_multi_seed(texts_train, struct_train, y, w, regime_b_cfg, ridge_fit_fn, alpha, n_rows)
        sc = brier_score(y, oof)
        if sc < best_score_b:
            best_score_b, best_alpha_b = sc, alpha
    print("regime B (ngram-only, heavier reg) chosen alpha:", best_alpha_b, "naive-cv brier:", best_score_b)

    lgb_candidates = [
        dict(n_estimators=200, num_leaves=8, min_child_samples=15, learning_rate=0.05,
             subsample=0.8, colsample_bytree=0.7, reg_lambda=0.5),
        dict(n_estimators=400, num_leaves=6, min_child_samples=25, learning_rate=0.03,
             subsample=0.8, colsample_bytree=0.6, reg_lambda=1.0),
    ]
    xgb_candidates = [
        dict(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
             colsample_bytree=0.6, reg_lambda=2.0),
        dict(n_estimators=350, max_depth=3, learning_rate=0.03, subsample=0.7,
             colsample_bytree=0.5, reg_lambda=3.0),
    ]

    best_lgb, best_lgb_score = None, np.inf
    for params in lgb_candidates:
        oof = oof_multi_seed(texts_train, struct_train, y, w, regime_a_cfg, fit_predict_lgb, params, n_rows)
        sc = brier_score(y, oof)
        if sc < best_lgb_score:
            best_lgb_score, best_lgb = sc, params
    print("chosen lgb params:", best_lgb, "naive-cv brier:", best_lgb_score)

    best_xgb, best_xgb_score = None, np.inf
    for params in xgb_candidates:
        oof = oof_multi_seed(texts_train, struct_train, y, w, regime_a_cfg, fit_predict_xgb, params, n_rows)
        sc = brier_score(y, oof)
        if sc < best_xgb_score:
            best_xgb_score, best_xgb = sc, params
    print("chosen xgb params:", best_xgb, "naive-cv brier:", best_xgb_score)

    oof_ra = oof_multi_seed(texts_train, struct_train, y, w, regime_a_cfg, ridge_fit_fn, alpha_vec_a, n_rows)
    oof_rb = oof_multi_seed(texts_train, struct_train, y, w, regime_b_cfg, ridge_fit_fn, best_alpha_b, n_rows)
    oof_l = oof_multi_seed(texts_train, struct_train, y, w, regime_a_cfg, fit_predict_lgb, best_lgb, n_rows)
    oof_x = oof_multi_seed(texts_train, struct_train, y, w, regime_a_cfg, fit_predict_xgb, best_xgb, n_rows)
    print("oof regime-A ridge brier:", brier_score(y, oof_ra))
    print("oof regime-B ridge brier:", brier_score(y, oof_rb))
    print("oof lgb brier:", brier_score(y, oof_l))
    print("oof xgb brier:", brier_score(y, oof_x))

    hedge_weights, hedge_score = search_blend_weights([oof_ra, oof_rb], y)
    print("chosen hedge weights (regime-A ridge, regime-B ridge):", hedge_weights, "hedge brier:", hedge_score)
    oof_hedge = normalize_probs(hedge_weights[0] * oof_ra + hedge_weights[1] * oof_rb)

    weights, blend_score = search_blend_weights([oof_hedge, oof_l, oof_x], y)
    print("chosen blend weights (ridge-hedge, lgb, xgb):", weights, "blend brier:", blend_score)

    oof_blend = normalize_probs(weights[0] * oof_hedge + weights[1] * oof_l + weights[2] * oof_x)

    floors = [0.0]
    best_floor, best_floor_score = 0.0, brier_score(y, oof_blend)
    for floor in floors:
        floored = apply_prob_floor(oof_blend, floor)
        sc = brier_score(y, floored)
        if sc < best_floor_score:
            best_floor_score, best_floor = sc, floor
    print("chosen probability floor:", best_floor, "oof brier:", best_floor_score)

    oof_final = apply_prob_floor(oof_blend, best_floor)

    oof_max_p = oof_final.max(axis=1)
    oof_max_y = y.max(axis=1)
    conf_calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    conf_calibrator.fit(oof_max_p, oof_max_y)
    calibrated_conf = conf_calibrator.predict(oof_max_p)
    print("mean abs conf error raw:", np.mean(np.abs(oof_max_p - oof_max_y)))
    print("mean abs conf error calibrated:", np.mean(np.abs(calibrated_conf - oof_max_y)))

    groups = build_opening_groups(texts_train)
    gsplits = grouped_splits(n_rows, groups, n_splits=N_SPLITS, seed=SEED)
    oof_group_diag = oof_for_config_and_model(texts_train, struct_train, y, w, regime_a_cfg, ridge_fit_fn, alpha_vec_a, gsplits)
    print("diagnostic (not used for selection) group-cv brier of regime-A config:", brier_score(y, oof_group_diag))

    Xtr_a, Xte_a = build_feature_block(texts_train, texts_test, struct_train, struct_test, regime_a_cfg)
    Xtr_b, Xte_b = build_feature_block(texts_train, texts_test, struct_train, struct_test, regime_b_cfg)

    pred_ra = normalize_probs(fit_predict_ridge(alpha_vec_a, Xtr_a, y, w, Xte_a))
    pred_rb = normalize_probs(fit_predict_ridge(best_alpha_b, Xtr_b, y, w, Xte_b))
    pred_l = normalize_probs(fit_predict_lgb(best_lgb, Xtr_a, y, w, Xte_a))
    pred_x = normalize_probs(fit_predict_xgb(best_xgb, Xtr_a, y, w, Xte_a))

    pred_hedge = normalize_probs(hedge_weights[0] * pred_ra + hedge_weights[1] * pred_rb)
    final_pred = normalize_probs(weights[0] * pred_hedge + weights[1] * pred_l + weights[2] * pred_x)
    final_pred = apply_prob_floor(final_pred, best_floor)
    final_max_p = final_pred.max(axis=1)
    final_conf = np.clip(conf_calibrator.predict(final_max_p), 0.0, 1.0)

    out = pd.DataFrame({
        "id": test["id"].values,
        "white_win_prob": final_pred[:, 0],
        "draw_prob": final_pred[:, 1],
        "black_win_prob": final_pred[:, 2],
        "confidence": final_conf,
    })

    out_dir = "working"
    os.makedirs(out_dir, exist_ok=True)
    archive_existing_submission(out_dir)
    out_path = os.path.join(out_dir, "submission.csv")
    out.to_csv(out_path, index=False)
    print("saved:", out_path, "rows:", len(out))


if __name__ == "__main__":
    main()
