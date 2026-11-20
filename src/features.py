"""Transforma a série do dólar em tabela para os modelos (features X e alvo y)."""
from __future__ import annotations

import pandas as pd

FEATURES = [
    "ret_1", "ret_2", "ret_3", "ret_5", "ret_10",
    "dist_media_5", "dist_media_20", "volat_10", "volat_30", "selic",
]


def construir_features(dolar: pd.DataFrame, selic: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por dia com features calculadas SÓ com dados até aquele dia."""
    df = dolar[["data", "valor"]].copy().sort_values("data").reset_index(drop=True)
    r = df["valor"].pct_change()
    for k in (1, 2, 3, 5, 10):
        df[f"ret_{k}"] = df["valor"].pct_change(k)
    df["dist_media_5"] = df["valor"] / df["valor"].rolling(5).mean() - 1
    df["dist_media_20"] = df["valor"] / df["valor"].rolling(20).mean() - 1
    df["volat_10"] = r.rolling(10).std()
    df["volat_30"] = r.rolling(30).std()
    # Selic vigente em cada dia (último valor conhecido até aquela data)
    sel = selic[["data", "valor"]].rename(columns={"valor": "selic"}).sort_values("data")
    df = pd.merge_asof(df, sel, on="data")
    # alvo: retorno do PRÓXIMO dia útil
    df["alvo_ret"] = df["valor"].shift(-1) / df["valor"] - 1
    df["alvo_sobe"] = (df["alvo_ret"] > 0).astype(int)
    return df


def linha_atual(df: pd.DataFrame) -> pd.DataFrame:
    """Última linha (hoje), que ainda não tem alvo, só features."""
    return df.dropna(subset=FEATURES).iloc[[-1]]


def treino_completo(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(subset=FEATURES + ["alvo_ret"]).reset_index(drop=True)


def proximos_dias_uteis(ultima: pd.Timestamp, n: int) -> list[pd.Timestamp]:
    # aproximação: ignora feriados (a PTAX não sai em feriados)
    return list(pd.bdate_range(ultima + pd.Timedelta(days=1), periods=n))
