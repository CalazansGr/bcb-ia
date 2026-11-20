"""Monta o corpus de treino do mini-LLM a partir dos dados do Banco Central.

Cada documento tem duas partes:

  <dados> data=19/11/2026 dolar=5.8889 dia=+1.37% sem=+0.45% mes=-2.10% selic=13.75 ipca=0.14 </dados>
  <texto> No pregão de 19/11/2026, o dólar subiu 1,37% e fechou a R$ 5,8889. ... </texto>¶

O modelo aprende a ESCREVER o texto a partir do registro de dados. Na hora de usar, damos só a parte
<dados> do dia mais recente (números reais do BC) e ele continua escrevendo.

Os textos de treino vêm de modelos de frase com variações (várias formas de dizer "subiu",
"caiu", etc.). Isso é de propósito: o mini-LLM aprende o estilo e a ligação números -> palavras.
Ele não raciocina sobre economia; só imita bem o padrão que viu.

Todos os NÚMEROS vêm de dados reais do Banco Central: nada é sorteado nem inventado. O que é escrito
por mim são só as frases-modelo (as palavras ao redor dos números). A validação usa dias reais que o
modelo nunca viu.
"""
from __future__ import annotations

import random

import pandas as pd

FIM = "¶"  # marca de fim de documento

# Faixas de variação diária (em %) que mudam o jeito de escrever:
LIMIAR_ESTAVEL = 0.10  # abaixo: "praticamente estável"
LIMIAR_NEUTRO = 0.25   # entre os dois: "variou pouco (+0,17%)", com o número; acima: sobe/cai
# (a faixa do meio evita um corte seco entre "estável" e "subiu", que confundia o modelo)


def montar_registros(dolar: pd.DataFrame, selic: pd.DataFrame, ipca: pd.DataFrame) -> pd.DataFrame:
    df = dolar[["data", "valor"]].rename(columns={"valor": "dolar"}).sort_values("data").reset_index(drop=True)
    df["dia"] = df["dolar"].pct_change() * 100
    df["sem"] = df["dolar"].pct_change(5) * 100
    df["mes"] = df["dolar"].pct_change(21) * 100
    s = selic[["data", "valor"]].rename(columns={"valor": "selic"}).sort_values("data")
    i = ipca[["data", "valor"]].rename(columns={"valor": "ipca"}).sort_values("data")
    df = pd.merge_asof(df, s, on="data")
    df = pd.merge_asof(df, i, on="data")
    return df.dropna().reset_index(drop=True)


def prompt_dados(r) -> str:
    return (f"<dados> data={r.data:%d/%m/%Y} dolar={r.dolar:.4f} dia={r.dia:+.2f}% sem={r.sem:+.2f}% "
            f"mes={r.mes:+.2f}% selic={r.selic:.2f} ipca={r.ipca:.2f} </dados>\n<texto> ")


def _br(x: float, casas: int) -> str:
    return f"{x:.{casas}f}".replace(".", ",")


def escrever_texto(r, rng: random.Random) -> str:
    """Texto de referência (professor) para um registro. É o que o mini-LLM vai aprender a imitar."""
    d, a = r.dia, abs(r.dia)
    preco = f"R$ {_br(r.dolar, 4)}"
    pct = f"{_br(a, 2)}%"
    sinal = "+" if d >= 0 else "-"
    abre = rng.choice([f"No pregão de {r.data:%d/%m/%Y}, ", f"Em {r.data:%d/%m/%Y}, ", f"Nesta data ({r.data:%d/%m/%Y}), "])

    if a < LIMIAR_ESTAVEL:
        f1 = rng.choice([f"o dólar ficou praticamente estável e fechou a {preco}",
                         f"a moeda americana teve variação desprezível e foi cotada a {preco}"])
    elif a < LIMIAR_NEUTRO:
        f1 = rng.choice([f"o dólar variou pouco ({sinal}{pct}) e fechou a {preco}",
                         f"a moeda americana oscilou {sinal}{pct} e foi cotada a {preco}"])
    elif d > 0:
        verbo = ("disparou" if a >= 1.0 else rng.choice(["subiu", "avançou", "teve alta de"]))
        f1 = (f"o dólar teve alta de {pct} e fechou a {preco}" if verbo == "teve alta de"
              else f"o dólar {verbo} {pct} e fechou a {preco}")
    else:
        verbo = ("despencou" if a >= 1.0 else rng.choice(["caiu", "recuou", "teve queda de"]))
        f1 = (f"o dólar teve queda de {pct} e fechou a {preco}" if verbo == "teve queda de"
              else f"o dólar {verbo} {pct} e fechou a {preco}")
    frases = [abre + f1 + "."]

    m = r.mes
    if abs(m) < 1.0:
        f2 = f"Em 21 dias úteis, a moeda ficou de lado ({'+' if m >= 0 else '-'}{_br(abs(m), 2)}%)."
    elif m > 0:
        f2 = rng.choice([f"No acumulado de 21 dias úteis, a alta chega a {_br(m, 2)}%.",
                         f"Em um mês, o dólar acumula valorização de {_br(m, 2)}%."])
    else:
        f2 = rng.choice([f"No acumulado de 21 dias úteis, a queda chega a {_br(abs(m), 2)}%.",
                         f"Em um mês, o dólar acumula desvalorização de {_br(abs(m), 2)}%."])
    frases.append(f2)

    if rng.random() < 0.8:
        frases.append(rng.choice([f"A meta da Selic está em {_br(r.selic, 2)}% ao ano.",
                                  f"A taxa Selic segue em {_br(r.selic, 2)}% ao ano."]))
    if rng.random() < 0.6:
        frases.append(rng.choice([f"O último IPCA mensal foi de {_br(r.ipca, 2)}%.",
                                  f"A inflação mensal (IPCA) mais recente marcou {_br(r.ipca, 2)}%."]))
    return " ".join(frases) + " </texto>" + FIM


def montar_documentos(reg: pd.DataFrame, copias: int = 3, seed: int = 0) -> list[str]:
    """Cada registro vira `copias` documentos com redações diferentes (mais variedade para aprender)."""
    rng = random.Random(seed)
    docs = []
    for r in reg.itertuples():
        for _ in range(copias):
            docs.append(prompt_dados(r) + escrever_texto(r, rng) + "\n")
    return docs
