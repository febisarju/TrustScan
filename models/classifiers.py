"""Text, metadata and stacked classifier implementations."""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from features.engineering import D

class TextLR:
    name = "LR-TFIDF"; thr = 0.5
    def __init__(self, C=5.0): self.C = C
    def fit(self, d, y, g=None):
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=100000,
                                   sublinear_tf=True, stop_words="english", dtype=np.float32)
        X = self.vec.fit_transform(d.text)
        self.clf = LogisticRegression(C=self.C, class_weight="balanced", solver="liblinear").fit(X, y)
        return self
    def logit(self, d): return self.clf.decision_function(self.vec.transform(d.text))
    def score(self, d): return self.clf.predict_proba(self.vec.transform(d.text))[:, 1]
    def top_terms(self, k=25):
        names = np.array(self.vec.get_feature_names_out()); w = self.clf.coef_[0]
        o = np.argsort(w)
        return list(zip(names[o[::-1][:k]], w[o[::-1][:k]])), list(zip(names[o[:k]], w[o[:k]]))

class TextSVM:
    name = "SVM-TFIDF"; thr = 0.0
    def fit(self, d, y, g=None):
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=100000,
                                   sublinear_tf=True, stop_words="english", dtype=np.float32)
        self.clf = LinearSVC(C=0.5, class_weight="balanced").fit(self.vec.fit_transform(d.text), y)
        return self
    def score(self, d): return self.clf.decision_function(self.vec.transform(d.text))

class DenseRF:
    name = "RF-Meta+Cues"; thr = 0.5
    def fit(self, d, y, g=None):
        self.clf = RandomForestClassifier(200, min_samples_leaf=2, class_weight="balanced_subsample",
                                          n_jobs=-1, random_state=0).fit(d.dense(), y)
        return self
    def score(self, d): return self.clf.predict_proba(d.dense())[:, 1]

def _xgb(y):
    spw = float(np.sqrt((y == 0).sum() / max((y == 1).sum(), 1)))
    return xgb.XGBClassifier(n_estimators=250, learning_rate=0.05, max_depth=4, subsample=0.8,
                             colsample_bytree=0.8, min_child_weight=2, reg_lambda=2.0,
                             scale_pos_weight=spw, tree_method="hist", n_jobs=-1,
                             eval_metric="aucpr", random_state=0, verbosity=0)

class DenseXGB:
    name = "XGB-Meta+Cues"; thr = 0.5
    def fit(self, d, y, g=None):
        self.clf = _xgb(y).fit(d.dense(), y); return self
    def score(self, d): return self.clf.predict_proba(d.dense())[:, 1]

class Stack:
    thr = 0.5
    def __init__(self, use_cue=True, inner=3, seed=0, name="TrustScan-Stack"):
        self.use_cue, self.inner, self.seed, self.name = use_cue, inner, seed, name
    def _feats(self, d, tl):
        X = d.dense(cue=self.use_cue, meta=True).copy()
        X.insert(0, "text_logit", tl); return X
    def fit(self, d, y, g):
        mk = lambda: LogisticRegression(C=5.0, class_weight="balanced", solver="liblinear")
        oof = np.zeros(len(y), dtype=float)
        cv = StratifiedGroupKFold(self.inner, shuffle=True, random_state=self.seed)
        for tr, va in cv.split(np.zeros(len(y)), y, g):
            fold_vec = TfidfVectorizer(
                ngram_range=(1, 2), min_df=3, max_features=100000,
                sublinear_tf=True, stop_words="english", dtype=np.float32)
            X_tr = fold_vec.fit_transform(d.text[tr])
            X_va = fold_vec.transform(d.text[va])
            oof[va] = mk().fit(X_tr, y[tr]).decision_function(X_va)

        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=100000,
                              sublinear_tf=True, stop_words="english", dtype=np.float32)
        X = vec.fit_transform(d.text)
        self.text = TextLR(); self.text.vec = vec; self.text.clf = mk().fit(X, y)
        self.clf = _xgb(y).fit(self._feats(d, oof), y)
        return self
    def score(self, d): return self.clf.predict_proba(self._feats(d, self.text.logit(d)))[:, 1]
    def explain(self, d, k=6, n_terms=6):
        X = self._feats(d, self.text.logit(d))
        contrib = self.clf.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
        tv = self.text.vec.transform(d.text).multiply(self.text.clf.coef_[0]).tocsr()
        names = np.array(self.text.vec.get_feature_names_out()); out = []
        for i in range(len(d)):
            o = np.argsort(contrib[i, :-1])[::-1][:k]
            feats = [(X.columns[j], float(X.iloc[i, j]), float(contrib[i, j])) for j in o if contrib[i, j] > 0]
            row = tv.getrow(i); idx = row.indices[np.argsort(row.data)[::-1][:n_terms]]
            out.append({"features": feats, "terms": [t for t, v in zip(names[idx], np.sort(row.data)[::-1][:n_terms]) if v > 0]})
        return out

def make_models():
    return {"LR-TFIDF": TextLR, "SVM-TFIDF": TextSVM, "RF-Meta+Cues": DenseRF, "XGB-Meta+Cues": DenseXGB,
            "Stack-noCues": lambda: Stack(use_cue=False, name="Stack-noCues"), "TrustScan-Stack": Stack}