"""Treino, avaliação e geração do mini-LLM.

Todos os números vêm de dados reais do Banco Central. Depois que o modelo escreve um texto, um
VERIFICADOR confere se os números citados batem com os dados; se não baterem, o texto é descartado e o
modelo escreve de novo (até `tentativas` vezes). Nada é inventado nem "corrigido" à mão.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict
from pathlib import Path

import torch

from src import bcb

from .corpus import FIM, LIMIAR_ESTAVEL, LIMIAR_NEUTRO, _br, montar_documentos, montar_registros, prompt_dados
from .modelo import Config, MiniGPT

CKPT = Path(__file__).resolve().parent.parent / "modelos" / "minillm.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TENTATIVAS = 5


def _registros():
    return montar_registros(bcb.carregar("dolar"), bcb.carregar("selic"), bcb.carregar("ipca"))


_SOBE = ("subiu", "avançou", "alta de", "disparou")
_CAI = ("caiu", "recuou", "queda de", "despencou")


def checar(texto: str, r) -> bool:
    """O texto está de acordo com os dados reais? Confere:
    - os NÚMEROS: preço, variação do dia (quando citada), do mês, Selic e IPCA (quando citados);
    - o SENTIDO das palavras: não pode dizer "subiu" se caiu (nem "alta" no mês se foi queda), e vice-versa.
    """
    ok = f"R$ {_br(r.dolar, 4)}" in texto
    if "Selic" in texto:
        ok &= f"{_br(r.selic, 2)}%" in texto
    if "IPCA" in texto:
        ok &= f"{_br(r.ipca, 2)}%" in texto

    # variação do dia (primeira frase do texto)
    frase1, a = texto.split(". ")[0], abs(r.dia)
    if a >= LIMIAR_NEUTRO:
        sobe, cai = any(p in frase1 for p in _SOBE), any(p in frase1 for p in _CAI)
        ok &= f"{_br(a, 2)}%" in frase1 and ((sobe and not cai) if r.dia > 0 else (cai and not sobe))
    elif a >= LIMIAR_ESTAVEL:
        ok &= f"{'+' if r.dia >= 0 else '-'}{_br(a, 2)}%" in frase1
    else:
        ok &= "estável" in frase1 or "desprezível" in frase1

    # variação do mês
    m = r.mes
    alta = "alta chega" in texto or "acumula valorização" in texto
    queda = "queda chega" in texto or "desvalorização" in texto
    if abs(m) < 1.0:
        ok &= f"{'+' if m >= 0 else '-'}{_br(abs(m), 2)}%" in texto
    else:
        ok &= f"{_br(abs(m), 2)}%" in texto and ((alta and not queda) if m > 0 else (queda and not alta))
    return bool(ok)


def _gerar_texto(modelo, stoi, itos, r, temperatura=0.5, top_k=20) -> str:
    prompt = prompt_dados(r)
    idx = torch.tensor([[stoi[c] for c in prompt]], device=DEVICE)
    modelo.eval()
    saida = modelo.gerar(idx, 600, temperatura, top_k, parar_em=stoi[FIM])
    return "".join(itos[i] for i in saida[0].tolist())[len(prompt):].replace("</texto>", "").replace(FIM, "").strip()


def escrever_confiavel(modelo, stoi, itos, r, tentativas: int = TENTATIVAS, temperatura: float = 0.5):
    """Gera até `tentativas` textos e devolve o primeiro que confere: (texto, nº da tentativa, conferiu?)."""
    t = ""
    for k in range(1, tentativas + 1):
        t = _gerar_texto(modelo, stoi, itos, r, temperatura)
        if checar(t, r):
            return t, k, True
    return t, tentativas, False


def treinar(passos: int = 2500, lote: int = 32, lr: float = 1e-3, avaliar_a_cada: int = 250) -> None:
    torch.manual_seed(0)
    reg = _registros()
    corte = int(len(reg) * 0.85)  # os 15% mais recentes ficam só para validação (dias nunca vistos)
    treino_docs = montar_documentos(reg.iloc[:corte], copias=3, seed=1)
    val_docs = montar_documentos(reg.iloc[corte:], copias=1, seed=2)

    chars = sorted(set("".join(montar_documentos(reg, copias=1, seed=3))) | set(FIM))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    enc = lambda docs: torch.tensor([stoi[c] for c in "".join(docs)], dtype=torch.long)
    dados_tr, dados_val = enc(treino_docs).to(DEVICE), enc(val_docs).to(DEVICE)

    cfg = Config(vocab=len(chars))
    modelo = MiniGPT(cfg).to(DEVICE)
    n_par = sum(p.numel() for p in modelo.parameters())
    print(f"Dispositivo: {DEVICE} | parâmetros: {n_par/1e6:.2f} M | vocabulário: {len(chars)} caracteres")
    print(f"Dados reais do BC: {len(reg)} dias ({reg['data'].iloc[0].date()} a {reg['data'].iloc[-1].date()})")
    print(f"Corpus: treino {len(dados_tr):,} caracteres | validação {len(dados_val):,}\n")

    def lote_de(dados):
        ix = torch.randint(len(dados) - cfg.bloco - 1, (lote,), device=DEVICE)
        x = torch.stack([dados[i:i + cfg.bloco] for i in ix])
        y = torch.stack([dados[i + 1:i + cfg.bloco + 1] for i in ix])
        return x, y

    @torch.no_grad()
    def perda_val() -> float:
        modelo.eval()
        ps = []
        for _ in range(20):
            x, y = lote_de(dados_val)
            with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
                ps.append(modelo(x, y)[1].item())
        modelo.train()
        return sum(ps) / len(ps)

    opt = torch.optim.AdamW(modelo.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    ini = time.time()
    for passo in range(1, passos + 1):
        # aquecimento + queda suave da taxa de aprendizado
        f = passo / 100 if passo < 100 else 0.5 * (1 + math.cos(math.pi * (passo - 100) / max(passos - 100, 1)))
        for g in opt.param_groups:
            g["lr"] = lr * max(f, 0.05)
        x, y = lote_de(dados_tr)
        with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
            _, perda = modelo(x, y)
        opt.zero_grad(set_to_none=True)
        perda.backward()
        torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
        opt.step()
        if passo % avaliar_a_cada == 0 or passo == 1:
            print(f"passo {passo:5d}/{passos} | perda treino {perda.item():.3f} | perda validação {perda_val():.3f} "
                  f"| {time.time()-ini:.0f}s", flush=True)
    print(f"\nTempo de treino: {time.time()-ini:.0f}s")

    CKPT.parent.mkdir(exist_ok=True)
    torch.save({"estado": modelo.state_dict(), "config": asdict(cfg), "chars": chars}, CKPT)
    print(f"Salvo em {CKPT}")

    _medir_fidelidade(modelo, stoi, itos, list(reg.iloc[corte:].itertuples()))


def _medir_fidelidade(modelo, stoi, itos, val) -> None:
    """Prova real: em dias que o modelo NUNCA viu, o texto está de acordo com os dados do BC?"""
    primeira = com_verificador = 0
    for r in val:
        _, k, ok = escrever_confiavel(modelo, stoi, itos, r)
        primeira += ok and k == 1
        com_verificador += ok
    n = len(val)
    print(f"\nFidelidade em {n} dias de validação (números E sentido das palavras conferem com os dados do BC):")
    print(f"  1ª tentativa .................. {primeira}/{n} ({primeira/n:.1%})")
    print(f"  com verificador (até {TENTATIVAS} tentativas) {com_verificador}/{n} ({com_verificador/n:.1%})")
    for r in val[-3:]:
        print(f"\n[{r.data:%d/%m/%Y}] {escrever_confiavel(modelo, stoi, itos, r)[0]}")


def avaliar_fidelidade() -> None:
    """Mede a fidelidade do modelo já treinado nos 15% de dias mais recentes (os que ele nunca viu)."""
    modelo, stoi, itos = carregar_modelo()
    reg = _registros()
    _medir_fidelidade(modelo, stoi, itos, list(reg.iloc[int(len(reg) * 0.85):].itertuples()))


def carregar_modelo():
    if not CKPT.exists():
        raise SystemExit("Modelo ainda não treinado. Rode: python run.py llm-treinar")
    ck = torch.load(CKPT, map_location=DEVICE, weights_only=True)
    modelo = MiniGPT(Config(**ck["config"])).to(DEVICE)
    modelo.load_state_dict(ck["estado"])
    stoi = {c: i for i, c in enumerate(ck["chars"])}
    return modelo, stoi, {i: c for c, i in stoi.items()}


def escrever(n: int = 3, temperatura: float = 0.7) -> None:
    """Escreve n textos sobre o dia MAIS RECENTE do Banco Central (dados atualizados agora)."""
    modelo, stoi, itos = carregar_modelo()
    r = list(_registros().itertuples())[-1]
    print(prompt_dados(r).split("\n")[0], "\n")
    for k in range(n):
        t, tent, ok = escrever_confiavel(modelo, stoi, itos, r, temperatura=temperatura)
        sel = "conferem com os dados" if ok else "NÃO conferem (descartar este texto)"
        print(f"[{k+1}] {t}\n    números {sel} (tentativa {tent} de {TENTATIVAS})\n")
