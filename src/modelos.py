"""A escadinha de algoritmos, do mais simples ao mais complexo, no mesmo problema:
prever o retorno do dólar no próximo dia útil.

IMPORTANTE: o split treino/teste é TEMPORAL (passado treina, futuro testa).
Embaralhar séries temporais vaza o futuro para o treino e gera resultados falsos.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from .features import FEATURES, treino_completo

SEED = 42


def criar_modelos() -> dict:
    """Nome -> modelo (regressores do retorno do dia seguinte)."""
    return {
        "1. regressao_linear": LinearRegression(),
        "3. arvore_decisao": DecisionTreeRegressor(max_depth=4, min_samples_leaf=30, random_state=SEED),
        "4. floresta_aleatoria": RandomForestRegressor(
            n_estimators=200, min_samples_leaf=20, n_jobs=-1, random_state=SEED),
        "5. gradient_boosting": GradientBoostingRegressor(
            n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=SEED),
        "6. knn": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5)),
        # o alvo (retorno diário, ~0,005) é minúsculo: a rede precisa dele padronizado para convergir
        "8. rede_neural": TransformedTargetRegressor(
            regressor=make_pipeline(
                StandardScaler(),
                MLPRegressor(hidden_layer_sizes=(32, 16), alpha=1e-2, max_iter=1000,
                             early_stopping=True, random_state=SEED)),
            transformer=StandardScaler()),
    }


def dividir_temporal(df: pd.DataFrame, frac_teste: float = 0.2):
    d = treino_completo(df)
    corte = int(len(d) * (1 - frac_teste))
    return d.iloc[:corte], d.iloc[corte:]


def avaliar(df: pd.DataFrame) -> pd.DataFrame:
    """Compara todos os modelos contra duas referências bobas. Quem não bate o 'ingenuo' não serve."""
    treino, teste = dividir_temporal(df)
    Xtr, ytr = treino[FEATURES], treino["alvo_ret"]
    Xte, yte = teste[FEATURES], teste["alvo_ret"]
    sobe_te = (yte > 0).astype(int)

    linhas = []

    def registrar(nome, pred):
        acerto_dir = float(np.mean((pred > 0) == (yte > 0))) if np.ptp(pred) > 0 else float("nan")
        linhas.append({
            "modelo": nome,
            "erro_medio_%": mean_absolute_error(yte, pred) * 100,
            "acerta_direcao_%": acerto_dir * 100,
        })

    # referências
    registrar("0a. ingenuo (amanha = hoje)", np.zeros(len(yte)))
    registrar("0b. media_historica", np.full(len(yte), ytr.mean()))

    for nome, modelo in criar_modelos().items():
        modelo.fit(Xtr, ytr)
        registrar(nome, modelo.predict(Xte))

    # logística: prevê SÓ a direção (sobe/desce)
    log = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    log.fit(Xtr, treino["alvo_sobe"])
    acc = accuracy_score(sobe_te, log.predict(Xte)) * 100
    linhas.append({"modelo": "2. regressao_logistica (sobe/desce)", "erro_medio_%": np.nan, "acerta_direcao_%": acc})
    linhas.append({"modelo": "0c. sempre_aposta_que_sobe", "erro_medio_%": np.nan,
                   "acerta_direcao_%": float(sobe_te.mean()) * 100})

    res = pd.DataFrame(linhas).sort_values("modelo").reset_index(drop=True)
    res.attrs["periodo_teste"] = (teste["data"].iloc[0].date(), teste["data"].iloc[-1].date(), len(teste))
    return res


def regimes(df: pd.DataFrame, k: int = 3):
    """K-means (sem rótulos): agrupa os dias em 'regimes' de mercado por volatilidade e tendência."""
    d = treino_completo(df)
    cols = ["volat_30", "ret_10", "dist_media_20"]
    esc = StandardScaler().fit(d[cols])
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(esc.transform(d[cols]))
    d = d.assign(regime=km.labels_)
    resumo = d.groupby("regime").agg(
        dias=("data", "count"),
        volatilidade_diaria_pct=("volat_30", lambda s: s.mean() * 100),
        retorno_10d_pct=("ret_10", lambda s: s.mean() * 100),
    )
    # nomeia do mais calmo ao mais agitado
    ordem = resumo["volatilidade_diaria_pct"].sort_values().index
    nomes = dict(zip(ordem, ["calmo", "intermediario", "agitado"][:k] if k <= 3 else range(k)))
    resumo.index = [nomes[i] for i in resumo.index]
    d["regime_nome"] = d["regime"].map(nomes)
    hoje = d.iloc[-1]
    return resumo.sort_values("volatilidade_diaria_pct"), hoje["regime_nome"], hoje["data"]
