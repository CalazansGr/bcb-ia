"""Mini-GPT escrito do zero em PyTorch (decoder-only transformer, no estilo nanoGPT).

Peças: embedding de caracteres + embedding de posição -> N blocos [atenção causal + MLP] -> projeção
para o vocabulário. O modelo só sabe fazer uma coisa: dado o texto até aqui, prever o PRÓXIMO caractere.
Gerar texto é repetir isso: prevê, sorteia, anexa, prevê de novo.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab: int
    bloco: int = 512      # quantos caracteres o modelo enxerga de uma vez
    camadas: int = 4
    cabecas: int = 4
    dim: int = 256
    dropout: float = 0.1


class Atencao(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.qkv = nn.Linear(c.dim, 3 * c.dim)
        self.saida = nn.Linear(c.dim, c.dim)
        self.cabecas, self.dropout = c.cabecas, c.dropout

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.cabecas, C // self.cabecas).transpose(1, 2) for t in (q, k, v))
        # is_causal=True: cada posição só olha para trás (não "cola" no futuro)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.dropout if self.training else 0.0)
        return self.saida(y.transpose(1, 2).contiguous().view(B, T, C))


class Bloco(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(c.dim), nn.LayerNorm(c.dim)
        self.atencao = Atencao(c)
        self.mlp = nn.Sequential(nn.Linear(c.dim, 4 * c.dim), nn.GELU(),
                                 nn.Linear(4 * c.dim, c.dim), nn.Dropout(c.dropout))

    def forward(self, x):
        x = x + self.atencao(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class MiniGPT(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.tok = nn.Embedding(c.vocab, c.dim)
        self.pos = nn.Embedding(c.bloco, c.dim)
        self.drop = nn.Dropout(c.dropout)
        self.blocos = nn.ModuleList(Bloco(c) for _ in range(c.camadas))
        self.ln = nn.LayerNorm(c.dim)
        self.cabeca = nn.Linear(c.dim, c.vocab, bias=False)
        self.cabeca.weight = self.tok.weight  # amarra pesos de entrada e saída
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, idx, alvo=None):
        B, T = idx.shape
        x = self.drop(self.tok(idx) + self.pos(torch.arange(T, device=idx.device)))
        for b in self.blocos:
            x = b(x)
        logits = self.cabeca(self.ln(x))
        perda = None if alvo is None else F.cross_entropy(logits.view(-1, logits.size(-1)), alvo.view(-1))
        return logits, perda

    @torch.no_grad()
    def gerar(self, idx, max_novos: int, temperatura: float = 0.7, top_k: int = 20, parar_em: int | None = None):
        for _ in range(max_novos):
            ctx = idx[:, -self.c.bloco:]
            logits, _ = self(ctx)
            logits = logits[:, -1, :] / max(temperatura, 1e-5)
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            prox = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            idx = torch.cat([idx, prox], dim=1)
            if parar_em is not None and prox.item() == parar_em:
                break
        return idx
