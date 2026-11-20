"""Coleta de dados do Banco Central (API SGS), sempre atualizada.

Cada execução baixa só o que falta desde o último dado salvo em data/ (cache CSV).
Se a internet cair, usa o cache e avisa.
"""
from __future__ import annotations

import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

# código SGS do Banco Central -> nome amigável
SERIES = {
    "dolar": 1,    # Dólar (PTAX) - venda, diário
    "selic": 432,  # Meta da taxa Selic, % a.a. (muda a cada reunião do Copom)
    "ipca": 433,   # IPCA, variação mensal %
}
# O dólar flutua desde 1999 e a série SGS só é comparável com o Real a partir de jul/1994
# (antes é outra moeda). Começamos em 2000 para usar só dados reais e comparáveis.
INICIO_PADRAO = date(2000, 1, 3)
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}/dados"


def _baixar_trecho(codigo: int, ini: date, fim: date) -> pd.DataFrame:
    params = {
        "formato": "json",
        "dataInicial": ini.strftime("%d/%m/%Y"),
        "dataFinal": fim.strftime("%d/%m/%Y"),
    }
    # A API do BC às vezes devolve resposta vazia/instável: tenta de novo com pausa crescente.
    for tentativa in range(1, 5):
        try:
            r = requests.get(URL.format(codigo=codigo), params=params, timeout=30)
            r.raise_for_status()
            dados = r.json()
            break
        except (requests.RequestException, ValueError):
            if tentativa == 4:
                raise
            time.sleep(2 * tentativa)
    if not dados:
        return pd.DataFrame(columns=["data", "valor"])
    df = pd.DataFrame(dados)
    df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    return df.dropna()


def _baixar(codigo: int, ini: date, fim: date) -> pd.DataFrame:
    # A API limita consultas longas em séries diárias: busca em blocos de 5 anos.
    partes, atual = [], ini
    while atual <= fim:
        fim_bloco = min(date(atual.year + 5, atual.month, 1) - timedelta(days=1), fim)
        partes.append(_baixar_trecho(codigo, atual, fim_bloco))
        atual = fim_bloco + timedelta(days=1)
    return pd.concat(partes, ignore_index=True) if partes else pd.DataFrame(columns=["data", "valor"])


def carregar(nome: str, atualizar: bool = True, desde: date = INICIO_PADRAO) -> pd.DataFrame:
    """Devolve DataFrame(data, valor) da série, atualizando o cache se possível."""
    if nome not in SERIES:
        raise ValueError(f"Série desconhecida: {nome}. Opções: {list(SERIES)}")
    DATA_DIR.mkdir(exist_ok=True)
    arq = DATA_DIR / f"{nome}.csv"
    cache = pd.read_csv(arq, parse_dates=["data"]) if arq.exists() else None

    if atualizar:
        hoje = date.today()
        cache_salvo = cache  # reserva: usada se o download falhar
        # cache que começa depois do início pedido (ex.: baixado numa versão antiga): baixa tudo de novo
        if cache is not None and cache["data"].min() > pd.Timestamp(desde) + pd.Timedelta(days=45):
            cache = None
        # recua 10 dias para pegar eventuais revisões do BC
        ini = (cache["data"].max().date() - timedelta(days=10)) if cache is not None else desde
        try:
            novo = _baixar(SERIES[nome], ini, hoje)
            cache = novo if cache is None else pd.concat([cache, novo], ignore_index=True)
            cache = cache.drop_duplicates("data", keep="last").sort_values("data")
            cache.to_csv(arq, index=False)
        except (requests.RequestException, ValueError) as e:
            cache = cache_salvo
            if cache is None:
                raise RuntimeError(f"Não consegui baixar '{nome}' do Banco Central e não há cache: {e}") from e
            print(f"[aviso] Não consegui atualizar '{nome}' ({type(e).__name__}); usando o cache salvo.")

    # a série da Selic traz datas futuras de reuniões; descarta o que ainda não aconteceu
    cache = cache[cache["data"] <= pd.Timestamp(date.today())]
    return cache.reset_index(drop=True)


def atualizar_tudo() -> dict[str, pd.DataFrame]:
    return {nome: carregar(nome) for nome in SERIES}


if __name__ == "__main__":
    for nome, df in atualizar_tudo().items():
        print(f"{nome:6s} {len(df):5d} linhas | último: {df['data'].iloc[-1].date()} = {df['valor'].iloc[-1]}")
