"""Chat interativo: você pergunta em português, o Claude chama as ferramentas (dados do BC e
modelos) e responde com números reais e atualizados.

Precisa de ANTHROPIC_API_KEY (no arquivo .env ou nas variáveis de ambiente).
"""
from __future__ import annotations

import json
import os

import anthropic
from anthropic import beta_tool
from dotenv import load_dotenv

from . import bcb
from .features import construir_features
from .modelos import avaliar, regimes
from .previsao import MODELOS_PREVISAO, prever

MODELO_CLAUDE = "claude-opus-5-5"

SYSTEM = """Você é um assistente de análise de dados macroeconômicos brasileiros (Banco Central: dólar PTAX, \
Selic meta, IPCA). Use SEMPRE as ferramentas para obter números; nunca invente valores.
Regras:
- Previsões são estatísticas e incertas. Sempre mostre a faixa (mínimo/máximo) e diga que o câmbio de curto \
prazo é muito difícil de prever.
- Antes de dar muita confiança a um modelo, consulte comparar_modelos: se ele não bate o 'ingenuo' \
(amanhã = hoje), diga isso com clareza.
- Isto é um projeto educacional, não é recomendação de investimento.
- Responda em português do Brasil, curto e direto, com datas explícitas."""

_cache: dict = {}


def _dados():
    if not _cache:
        _cache["dolar"] = bcb.carregar("dolar")
        _cache["selic"] = bcb.carregar("selic")
        _cache["ipca"] = bcb.carregar("ipca")
    return _cache


@beta_tool
def ultimos_valores(serie: str, n: int = 5) -> str:
    """Últimos valores de uma série do Banco Central.

    Args:
        serie: 'dolar' (PTAX venda, R$), 'selic' (meta, % a.a.) ou 'ipca' (variação mensal, %).
        n: quantos pontos mais recentes (1 a 60).
    """
    df = _dados()[serie].tail(max(1, min(int(n), 60)))
    return json.dumps([{"data": str(d.date()), "valor": float(v)} for d, v in zip(df["data"], df["valor"])])


@beta_tool
def estatisticas(serie: str, dias: int = 365) -> str:
    """Resumo estatístico (mínimo, máximo, média, variação) de uma série numa janela de dias corridos.

    Args:
        serie: 'dolar', 'selic' ou 'ipca'.
        dias: tamanho da janela em dias corridos (ex.: 30, 365, 1825).
    """
    df = _dados()[serie]
    j = df[df["data"] >= df["data"].iloc[-1] - __import__("pandas").Timedelta(days=int(dias))]
    return json.dumps({
        "de": str(j["data"].iloc[0].date()), "ate": str(j["data"].iloc[-1].date()), "pontos": len(j),
        "minimo": float(j["valor"].min()), "maximo": float(j["valor"].max()),
        "media": float(j["valor"].mean()), "primeiro": float(j["valor"].iloc[0]),
        "ultimo": float(j["valor"].iloc[-1]),
        "variacao_pct": float((j["valor"].iloc[-1] / j["valor"].iloc[0] - 1) * 100),
    })


@beta_tool
def prever_dolar(dias: int = 1, modelo: str = "regressao_linear") -> str:
    """Prevê o dólar PTAX para os próximos dias úteis, com faixa de ~95%.

    Args:
        dias: quantos dias úteis à frente (1 a 30).
        modelo: media_historica, regressao_linear, arvore_decisao, floresta_aleatoria,
            gradient_boosting, knn ou rede_neural.
    """
    d = _dados()
    if modelo not in MODELOS_PREVISAO:
        return f"Modelo inválido. Opções: {MODELOS_PREVISAO}"
    p = prever(d["dolar"], d["selic"], dias, modelo)
    p["data"] = p["data"].astype(str)
    return json.dumps({"modelo": modelo, "previsoes": p.round(4).to_dict("records")})


@beta_tool
def comparar_modelos() -> str:
    """Compara todos os modelos num teste fora da amostra (últimos 20% do histórico) contra o 'ingenuo'."""
    d = _dados()
    res = avaliar(construir_features(d["dolar"], d["selic"]))
    ini, fim, n = res.attrs["periodo_teste"]
    return json.dumps({"periodo_teste": f"{ini} a {fim} ({n} dias)", "resultados": res.round(3).to_dict("records")})


@beta_tool
def regime_atual() -> str:
    """Diz em qual 'regime' de mercado o dólar está hoje (calmo, intermediário ou agitado), via k-means."""
    d = _dados()
    resumo, atual, data = regimes(construir_features(d["dolar"], d["selic"]))
    return json.dumps({"data": str(data.date()), "regime_atual": atual, "regimes": resumo.round(3).reset_index().to_dict("records")})


def main() -> None:
    load_dotenv()
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Faltou a ANTHROPIC_API_KEY. Copie .env.example para .env e preencha a chave.")
        return
    client = anthropic.Anthropic()
    print("Atualizando dados do Banco Central...")
    _dados()
    print("Pronto. Pergunte algo (ex.: 'quanto o dólar fecha amanhã?'). Digite 'sair' para encerrar.\n")

    historico: list[dict] = []
    while True:
        try:
            pergunta = input("você> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if pergunta.lower() in {"sair", "exit", "quit"}:
            break
        if not pergunta:
            continue
        historico.append({"role": "user", "content": pergunta})
        runner = client.beta.messages.tool_runner(
            model=MODELO_CLAUDE,
            max_tokens=16000,
            system=SYSTEM,
            tools=[ultimos_valores, estatisticas, prever_dolar, comparar_modelos, regime_atual],
            messages=list(historico),
        )
        ultima = None
        for ultima in runner:
            pass
        if ultima is None or ultima.stop_reason == "refusal":
            print("ia> (o modelo recusou esta pergunta)\n")
            historico.pop()
            continue
        texto = "".join(b.text for b in ultima.content if b.type == "text")
        print(f"ia> {texto}\n")
        historico.append({"role": "assistant", "content": texto})


if __name__ == "__main__":
    main()
