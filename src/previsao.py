"""Previsão do dólar para os próximos dias úteis, com faixa de incerteza.

Previsão de vários dias é recursiva: prevê o dia seguinte, empurra o resultado para a série,
recalcula as features e prevê de novo. O erro se acumula, por isso a faixa abre com o tempo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import FEATURES, construir_features, linha_atual, proximos_dias_uteis, treino_completo
from .modelos import criar_modelos

MODELOS_PREVISAO = ["media_historica", "regressao_linear", "arvore_decisao",
                    "floresta_aleatoria", "gradient_boosting", "knn", "rede_neural"]


def _modelo_por_nome(nome: str):
    if nome == "media_historica":
        return None
    for chave, modelo in criar_modelos().items():
        if chave.split(". ", 1)[1] == nome:
            return modelo
    raise ValueError(f"Modelo desconhecido: {nome}. Opções: {MODELOS_PREVISAO}")


def prever(dolar: pd.DataFrame, selic: pd.DataFrame, dias: int = 5, modelo: str = "regressao_linear") -> pd.DataFrame:
    """Devolve DataFrame(data, previsao, minimo, maximo). Faixa ≈ 95% usando a volatilidade histórica."""
    dias = max(1, min(int(dias), 30))
    df = construir_features(dolar, selic)
    treino = treino_completo(df)
    ret_medio = treino["alvo_ret"].mean()
    # volatilidade do erro de 1 dia, medida nos últimos 2 anos
    sigma = treino["alvo_ret"].tail(500).std()

    mdl = _modelo_por_nome(modelo)
    if mdl is not None:
        mdl.fit(treino[FEATURES], treino["alvo_ret"])

    serie = dolar[["data", "valor"]].copy()
    previsoes = []
    datas = proximos_dias_uteis(serie["data"].iloc[-1], dias)
    for i, data in enumerate(datas, start=1):
        feats = linha_atual(construir_features(serie, selic))
        ret = ret_medio if mdl is None else float(mdl.predict(feats[FEATURES])[0])
        novo_valor = float(serie["valor"].iloc[-1]) * (1 + ret)
        serie = pd.concat([serie, pd.DataFrame({"data": [data], "valor": [novo_valor]})], ignore_index=True)
        largura = 1.96 * sigma * np.sqrt(i)
        previsoes.append({"data": data.date(), "previsao": novo_valor,
                          "minimo": novo_valor * (1 - largura), "maximo": novo_valor * (1 + largura)})
    return pd.DataFrame(previsoes)
