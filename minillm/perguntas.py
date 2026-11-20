"""Interpreta o que você digita e escolhe QUAL DIA o mini-LLM vai descrever.

IMPORTANTE: isto NÃO é inteligência artificial. São regras simples (expressões regulares) que reconhecem
alguns tipos de pedido em português. Quem escreve o texto continua sendo o mini-LLM; aqui só se decide o dia.
O programa sempre mostra como entendeu o pedido, e se não reconhecer, diz isso e lista o que sabe fazer.

Pedidos que reconhece:
  data exata        17/11/2026 | 2026-03-02 | 17 de novembro de 2026 | 17/11 | hoje | ontem | anteontem
  relativos         semana passada | mês passado | último dia | mais recente
  mês (e ano)       novembro de 2025 | março   -> último dia útil com dado daquele mês
  recordes          maior alta | maior queda | dólar mais caro | dólar mais barato
                    (em um período: de 2025 | este ano | ano passado | mês passado | outubro de 2024 |
                     últimos 30 dias; sem período, vale todo o histórico)
"""
from __future__ import annotations

import calendar
import re
import unicodedata
from bisect import bisect_right
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

DIAS = ["segunda-feira", "terça-feira", "quarta-feira", "quinta-feira", "sexta-feira", "sábado", "domingo"]
MESES = ["janeiro", "fevereiro", "marco", "abril", "maio", "junho", "julho", "agosto", "setembro", "outubro",
         "novembro", "dezembro"]
MESES_BR = ["janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho", "agosto", "setembro", "outubro",
            "novembro", "dezembro"]
# \b nos dois lados: sem isso, "maio" casaria dentro de "maior"
_RE_MES = r"\b(" + "|".join(MESES) + r")\b"
AJUDA = ("Não entendi o pedido. Eu reconheço:\n"
         "  - uma data: 17/11/2026, 2026-03-02, 17 de novembro de 2026, 17/11, hoje, ontem, anteontem\n"
         "  - relativos: semana passada, mês passado, último dia, mais recente\n"
         "  - um mês: novembro de 2025, março\n"
         "  - recordes: maior alta, maior queda, dólar mais caro, dólar mais barato\n"
         "    (com período opcional: de 2025, este ano, ano passado, mês passado, outubro de 2024, últimos 30 dias)")


@dataclass
class Escolha:
    registro: object | None  # linha de dados do dia escolhido (None se não deu para entender)
    explicacao: str = ""     # como o pedido foi entendido
    mensagem: str = ""       # erro ou ajuda, quando registro é None


class Base:
    """Os dias com dados reais, prontos para consulta."""

    def __init__(self, reg: pd.DataFrame):
        self.reg = reg
        self.por_data = {r.data.date(): r for r in reg.itertuples()}
        self.datas = sorted(self.por_data)
        self.primeiro, self.ultimo = self.datas[0], self.datas[-1]

    def ultimo_ate(self, d: date):
        """Último dia com dado em ou antes de d (None se não houver)."""
        i = bisect_right(self.datas, d)
        return self.datas[i - 1] if i else None

    def ultimo_no_periodo(self, ini: date, fim: date):
        d = self.ultimo_ate(fim)
        return d if d is not None and d >= ini else None

    def periodo(self, ini: date, fim: date) -> pd.DataFrame:
        return self.reg[(self.reg["data"] >= pd.Timestamp(ini)) & (self.reg["data"] <= pd.Timestamp(fim))]


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFD", s.lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn").strip()


def _br(x: float, casas: int = 2) -> str:
    return f"{x:.{casas}f}".replace(".", ",")


def _fmt_dia(d: date) -> str:
    return f"{d:%d/%m/%Y} ({DIAS[d.weekday()]})"


def _mes_range(ano: int, mes: int) -> tuple[date, date]:
    return date(ano, mes, 1), date(ano, mes, calendar.monthrange(ano, mes)[1])


def _escolha_data(base: Base, d: date, como: str) -> Escolha:
    """Dia exato; sem dado (fim de semana, feriado, futuro) usa o dia útil anterior."""
    if d < base.primeiro:
        return Escolha(None, mensagem=f"Os dados começam em {base.primeiro:%d/%m/%Y}.")
    if d > base.ultimo:
        return Escolha(base.por_data[base.ultimo],
                       f"Entendi: {como}. Ainda não há dado para {d:%d/%m/%Y}; usando o mais recente: "
                       f"{_fmt_dia(base.ultimo)}.")
    if d not in base.por_data:
        ant = base.ultimo_ate(d)
        return Escolha(base.por_data[ant],
                       f"Entendi: {como}. Sem cotação em {d:%d/%m/%Y}, {DIAS[d.weekday()]} "
                       f"(fim de semana ou feriado); usando o dia útil anterior: {_fmt_dia(ant)}.")
    return Escolha(base.por_data[d], f"Entendi: {como} -> {_fmt_dia(d)}.")


def _periodo(t: str, hoje: date, base: Base):
    """Acha um período no pedido. Devolve (inicio, fim, rotulo); sem período, todo o histórico."""
    if re.search(r"ano passado", t):
        return date(hoje.year - 1, 1, 1), date(hoje.year - 1, 12, 31), f"ano passado ({hoje.year - 1})"
    m = re.search(_RE_MES + r"(?: de| /)?\s*(\d{4})?", t)
    if m:
        mes = MESES.index(m[1]) + 1
        if m[2]:
            ano = int(m[2])
        else:  # mês sem ano: a ocorrência mais recente que já aconteceu
            ano = hoje.year if mes <= hoje.month else hoje.year - 1
        ini, fim = _mes_range(ano, mes)
        return ini, fim, f"{MESES_BR[mes - 1]} de {ano}"
    if re.search(r"mes passado", t):
        u = hoje.replace(day=1) - timedelta(days=1)
        ini, fim = _mes_range(u.year, u.month)
        return ini, fim, f"mês passado ({u.month:02d}/{u.year})"
    if re.search(r"(este|neste|esse|nesse) mes", t):
        ini, fim = _mes_range(hoje.year, hoje.month)
        return ini, hoje, f"este mês ({hoje.month:02d}/{hoje.year})"
    m = re.search(r"ultimos?\s+(\d{1,4})\s+dias", t)
    if m:
        return hoje - timedelta(days=int(m[1])), hoje, f"últimos {m[1]} dias"
    if re.search(r"(este|neste|esse|nesse) ano|no ano\b", t):
        return date(hoje.year, 1, 1), hoje, f"este ano ({hoje.year})"
    m = re.search(r"\b(19\d{2}|20\d{2})\b", t)
    if m:
        a = int(m[1])
        return date(a, 1, 1), date(a, 12, 31), f"ano {a}"
    return base.primeiro, base.ultimo, "todo o histórico"


# (padrão, coluna, 'max'/'min', nome)
_RECORDES = [
    (r"maior alta", "dia", "max", "maior alta diária"),
    (r"maior queda", "dia", "min", "maior queda diária"),
    (r"(mais caro|maxima|maximo|pico)", "dolar", "max", "dólar mais caro"),
    (r"(mais barato|minima|minimo)", "dolar", "min", "dólar mais barato"),
]


def interpretar(texto: str, base: Base, hoje: date) -> Escolha:
    t = _norm(texto)
    if not t:
        return Escolha(None, mensagem=AJUDA)

    # 1) recordes: "maior alta de 2025"
    for padrao, col, modo, nome in _RECORDES:
        if re.search(padrao, t):
            ini, fim, rotulo = _periodo(t, hoje, base)
            sub = base.periodo(ini, fim)
            if sub.empty:
                return Escolha(None, mensagem=f"Não há dados em {rotulo}.")
            linha = sub.loc[sub[col].idxmax() if modo == "max" else sub[col].idxmin()]
            d = linha["data"].date()
            valor = f"{'+' if linha['dia'] >= 0 else ''}{_br(linha['dia'])}%" if col == "dia" else f"R$ {_br(linha['dolar'], 4)}"
            return Escolha(base.por_data[d], f"Entendi: {nome}, {rotulo} -> {_fmt_dia(d)}, {valor}.")

    # 2) relativos
    if re.search(r"anteontem", t):
        return _escolha_data(base, hoje - timedelta(days=2), "anteontem")
    if re.search(r"semana passada", t):
        seg = hoje - timedelta(days=hoje.weekday() + 7)
        d = base.ultimo_no_periodo(seg, seg + timedelta(days=6))
        if d is None:
            return Escolha(None, mensagem="Não há dados na semana passada.")
        return Escolha(base.por_data[d], f"Entendi: semana passada ({seg:%d/%m} a {seg + timedelta(days=6):%d/%m}) "
                                         f"-> último dia útil dela: {_fmt_dia(d)}.")
    if re.search(r"mes passado", t):
        u = hoje.replace(day=1) - timedelta(days=1)
        ini, fim = _mes_range(u.year, u.month)
        d = base.ultimo_no_periodo(ini, fim)
        if d is None:
            return Escolha(None, mensagem="Não há dados no mês passado.")
        return Escolha(base.por_data[d], f"Entendi: mês passado -> último dia útil dele: {_fmt_dia(d)}.")
    if re.search(r"ultimo dia|mais recente|\brecente\b|\bagora\b|\batual\b", t):
        return Escolha(base.por_data[base.ultimo], f"Entendi: dia mais recente -> {_fmt_dia(base.ultimo)}.")

    # 3) data exata
    try:
        m = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", t)
        if m:
            return _escolha_data(base, date(int(m[3]), int(m[2]), int(m[1])), f"data {m[0]}")
        m = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", t)
        if m:
            return _escolha_data(base, date(int(m[1]), int(m[2]), int(m[3])), f"data {m[0]}")
        m = re.search(r"(\d{1,2}) de " + _RE_MES + r"(?: de)? (\d{4})", t)
        if m:
            mes = MESES.index(m[2]) + 1
            return _escolha_data(base, date(int(m[3]), mes, int(m[1])),
                                 f"data {int(m[1])} de {MESES_BR[mes - 1]} de {m[3]}")
        m = re.search(r"(\d{1,2}) de " + _RE_MES, t) or re.search(r"\b(\d{1,2})/(\d{1,2})\b(?!/)", t)
        if m:
            dia = int(m[1])
            mes = MESES.index(m[2]) + 1 if m[2] in MESES else int(m[2])
            d = date(hoje.year, mes, dia)
            if d > hoje:  # sem ano: a ocorrência mais recente que já passou
                d = date(hoje.year - 1, mes, dia)
            return _escolha_data(base, d, f"data {dia:02d}/{mes:02d} (sem ano, considerei {d.year})")
    except ValueError:
        return Escolha(None, mensagem="Essa data não existe no calendário.")
    if re.search(r"\bhoje\b", t):
        return _escolha_data(base, hoje, "hoje")
    if re.search(r"\bontem\b", t):
        return _escolha_data(base, hoje - timedelta(days=1), "ontem")

    # 4) um mês (e ano) sozinho: último dia útil dele
    if re.search(_RE_MES, t):
        ini, fim, rotulo = _periodo(t, hoje, base)
        d = base.ultimo_no_periodo(ini, fim)
        if d is None:
            return Escolha(None, mensagem=f"Não há dados em {rotulo}.")
        return Escolha(base.por_data[d], f"Entendi: {rotulo} -> último dia útil com dado: {_fmt_dia(d)}.")

    return Escolha(None, mensagem=AJUDA)
