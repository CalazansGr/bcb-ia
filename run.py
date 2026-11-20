"""Ponto de entrada.

  python run.py atualizar            # baixa/atualiza dados do Banco Central
  python run.py avaliar              # compara todos os algoritmos (teste fora da amostra)
  python run.py prever --dias 5      # previsão do dólar (--modelo regressao_linear, etc.)
  python run.py regimes              # k-means: em que "regime" o dólar está
  python run.py chat                 # conversa com a IA (precisa de ANTHROPIC_API_KEY)
  python run.py llm-treinar          # treina o mini-LLM do zero (GPU NVIDIA)
  python run.py llm-escrever         # o mini-LLM escreve sobre o dia mais recente do BC
  python run.py llm-escrever --data 17/11/2026   # ...ou sobre uma data específica
  python run.py llm-conversar        # você digita datas e ele escreve o boletim de cada uma
  python run.py llm-avaliar          # mede a fidelidade do mini-LLM em dias que ele nunca viu
"""
import argparse
import sys

import pandas as pd

# acentos corretos no terminal do Windows
sys.stdout.reconfigure(encoding="utf-8")

from src import bcb
from src.features import construir_features
from src.modelos import avaliar, regimes
from src.previsao import MODELOS_PREVISAO, prever

pd.set_option("display.width", 140)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("comando", choices=["atualizar", "avaliar", "prever", "regimes", "chat",
                                        "llm-treinar", "llm-escrever", "llm-avaliar", "llm-conversar"])
    ap.add_argument("--data", default=None, help="data do boletim do mini-LLM (ex.: 17/11/2026)")
    ap.add_argument("--dias", type=int, default=5)
    ap.add_argument("--passos", type=int, default=2500, help="passos de treino do mini-LLM")
    ap.add_argument("--modelo", default="regressao_linear", choices=MODELOS_PREVISAO)
    a = ap.parse_args()

    if a.comando == "chat":
        from src.chat import main as chat_main
        return chat_main()
    if a.comando == "llm-treinar":
        from minillm.treino import treinar
        return treinar(passos=a.passos)
    if a.comando == "llm-escrever":
        from minillm.treino import escrever
        return escrever(data=a.data)
    if a.comando == "llm-conversar":
        from minillm.treino import conversar
        return conversar()
    if a.comando == "llm-avaliar":
        from minillm.treino import avaliar_fidelidade
        return avaliar_fidelidade()

    dolar, selic = bcb.carregar("dolar"), bcb.carregar("selic")
    print(f"Dólar: {len(dolar)} dias, último = R$ {dolar['valor'].iloc[-1]:.4f} em {dolar['data'].iloc[-1].date()}")

    if a.comando == "atualizar":
        bcb.atualizar_tudo()
    elif a.comando == "avaliar":
        res = avaliar(construir_features(dolar, selic))
        ini, fim, n = res.attrs["periodo_teste"]
        print(f"\nTeste fora da amostra: {ini} a {fim} ({n} dias)\n")
        print(res.round(3).to_string(index=False))
        print("\nLeia assim: quem não tem erro MENOR que o '0a. ingenuo' não está agregando nada.")
    elif a.comando == "prever":
        p = prever(dolar, selic, a.dias, a.modelo)
        print(f"\nPrevisão ({a.modelo}), faixa ≈ 95%:\n")
        print(p.round(4).to_string(index=False))
        print("\nAviso: projeto educacional; câmbio de curto prazo é muito difícil de prever.")
    elif a.comando == "regimes":
        resumo, atual, data = regimes(construir_features(dolar, selic))
        print(resumo.round(3).to_string())
        print(f"\nRegime em {data.date()}: {atual}")


if __name__ == "__main__":
    main()
