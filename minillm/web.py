"""Interface web local do mini-LLM (python run.py web).

Usa só a biblioteca padrão do Python: nada novo para instalar. O servidor escuta APENAS em 127.0.0.1, ou seja,
só o seu computador acessa. Nada sai da sua máquina, a não ser o download dos dados do Banco Central.

Rotas:
  /               a página (minillm/pagina.html)
  /api/info       informações do modelo e dos dados (e confere dados novos a cada 30 min)
  /api/atualizar  busca agora, no Banco Central, dados mais novos que os salvos
  /api/boletim    ?pedido=...   interpreta o pedido, o mini-LLM escreve e o verificador confere
  /api/serie      ?centro=AAAA-MM-DD&dias=N   pontos do gráfico do dólar
"""
from __future__ import annotations

import json
import threading
import webbrowser
from datetime import date, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from src import bcb

from .perguntas import Base, interpretar
from .treino import DEVICE, TENTATIVAS, _registros, carregar_modelo, escrever_confiavel

PAGINA = Path(__file__).with_name("pagina.html")
MSG_OFFLINE = "Não consegui consultar o Banco Central agora; estou usando os dados já salvos."
EXEMPLOS = ["semana passada", "mês passado", "maior alta de 2025", "maior queda de 2020",
            "dólar mais caro de 2024", "novembro de 2025"]
# só aceita a página carregada por estes endereços (protege contra páginas de outros sites falando com o servidor)
CSP = "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'"


def _registro_json(r) -> dict:
    return {"data": r.data.date().isoformat(), "dolar": float(r.dolar), "dia": float(r.dia),
            "sem": float(r.sem), "mes": float(r.mes), "selic": float(r.selic), "ipca": float(r.ipca)}


def servir(porta: int = 8765, abrir: bool = True) -> None:
    print("Carregando o mini-LLM e os dados do Banco Central...")
    modelo, stoi, itos = carregar_modelo()
    estado = {"base": Base(_registros()), "verificado": datetime.now(), "erro": ""}
    if bcb.ERROS:  # já ao ligar, sem internet: avisa que está usando os dados salvos
        estado["erro"] = MSG_OFFLINE
    parametros = sum(p.numel() for p in modelo.parameters())
    trava = threading.Lock()  # um texto por vez
    trava_dados = threading.Lock()

    def atualizar(forcar: bool = False) -> dict:
        """Busca dados novos no Banco Central (só baixa o que falta). Sem `forcar`, no máximo a cada 30 minutos."""
        with trava_dados:
            if not forcar and datetime.now() - estado["verificado"] < timedelta(minutes=30):
                return {"mudou": False}
            antes = estado["base"].ultimo
            try:
                estado["base"] = Base(_registros())
                # o bcb.carregar não derruba o programa se a internet cair: usa o cache e registra a falha em ERROS
                estado["erro"] = MSG_OFFLINE if bcb.ERROS else ""
            except Exception as e:  # sem internet e sem cache
                estado["erro"] = f"Não consegui falar com o Banco Central ({e}). Usando os dados salvos."
            estado["verificado"] = datetime.now()
            return {"mudou": estado["base"].ultimo != antes, "antes": antes, "agora": estado["base"].ultimo}

    def lacuna(ultimo: date, hoje: date) -> list[dict]:
        """O que houve entre o último dia com cotação e hoje: explica por que não há dado mais novo."""
        itens, d = [], ultimo + timedelta(days=1)
        while d <= hoje and len(itens) < 8:
            if d.weekday() >= 5:
                motivo = "fim de semana"
            elif d == hoje:
                motivo = "cotação de hoje ainda não publicada"
            else:
                motivo = "feriado ou sem publicação"
            if itens and itens[-1]["motivo"] == motivo and itens[-1]["fim"] == d - timedelta(days=1):
                itens[-1]["fim"] = d
            else:
                itens.append({"ini": d, "fim": d, "motivo": motivo})
            d += timedelta(days=1)
        return [{"quando": f"{i['ini']:%d/%m}" + (f" a {i['fim']:%d/%m}" if i["fim"] != i["ini"] else ""),
                 "motivo": i["motivo"]} for i in itens]

    def info() -> dict:
        atualizar()
        base = estado["base"]
        return {"ok": True, "primeiro": base.primeiro.isoformat(), "ultimo": base.ultimo.isoformat(),
                "verificado": estado["verificado"].strftime("%H:%M"), "erro": estado["erro"],
                "lacuna": lacuna(base.ultimo, date.today()),
                "parametros": parametros, "dispositivo": "GPU (CUDA)" if DEVICE == "cuda" else "CPU",
                "exemplos": EXEMPLOS}

    def atualizar_agora() -> dict:
        r = atualizar(forcar=True)
        ultimo = estado["base"].ultimo
        if estado["erro"]:
            msg = estado["erro"]
        elif r["mudou"]:
            msg = f"Novos dados! O último dia com cotação agora é {ultimo:%d/%m/%Y} (antes: {r['antes']:%d/%m/%Y})."
        else:
            msg = f"Já estava atualizado: o Banco Central não publicou nada depois de {ultimo:%d/%m/%Y}."
        return {"ok": True, "mudou": bool(r["mudou"]), "mensagem": msg}

    def boletim(pedido: str) -> dict:
        atualizar()
        esc = interpretar(pedido.strip()[:200], estado["base"], date.today())
        if esc.registro is None:
            return {"ok": False, "mensagem": esc.mensagem}
        with trava:
            texto, tentativa, conferiu = escrever_confiavel(modelo, stoi, itos, esc.registro, temperatura=0.7)
        return {"ok": True, "explicacao": esc.explicacao, "registro": _registro_json(esc.registro),
                "texto": texto, "conferiu": bool(conferiu), "tentativa": tentativa, "tentativas": TENTATIVAS}

    def serie(centro: date, dias: int) -> dict:
        base = estado["base"]
        dias = max(30, min(dias, 9000))
        ini = centro - timedelta(days=dias)
        fim = min(centro + timedelta(days=max(dias // 4, 10)), base.ultimo)
        sub = base.periodo(ini, fim).reset_index(drop=True)
        if len(sub) < 2:
            return {"ok": False, "mensagem": "Sem dados nesse período."}
        passo = max(1, len(sub) // 900)  # no máximo ~900 pontos no gráfico
        idx = set(range(0, len(sub), passo)) | {len(sub) - 1}
        escolhido = sub.index[sub["data"].dt.date == centro]
        if len(escolhido):
            idx.add(int(escolhido[0]))
        sub = sub.loc[sorted(idx)]
        return {"ok": True, "escolhido": centro.isoformat(),
                "datas": [d.date().isoformat() for d in sub["data"]], "valores": [float(v) for v in sub["dolar"]]}

    class Handler(BaseHTTPRequestHandler):
        def _enviar(self, corpo: bytes, tipo: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", tipo)
            self.send_header("Content-Length", str(len(corpo)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy", CSP)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(corpo)

        def _json(self, obj: dict, status: int = 200) -> None:
            self._enviar(json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8", status)

        def do_GET(self) -> None:  # noqa: N802
            host = (self.headers.get("Host") or "").split(":")[0]
            if host not in ("127.0.0.1", "localhost"):
                return self._json({"ok": False, "mensagem": "Acesso negado."}, 403)
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                if u.path in ("/", "/index.html"):
                    return self._enviar(PAGINA.read_bytes(), "text/html; charset=utf-8")
                if u.path == "/api/info":
                    return self._json(info())
                if u.path == "/api/atualizar":
                    return self._json(atualizar_agora())
                if u.path == "/api/boletim":
                    return self._json(boletim(q.get("pedido", "")))
                if u.path == "/api/serie":
                    try:
                        centro = date.fromisoformat(q.get("centro", ""))
                        dias = int(q.get("dias", "365"))
                    except ValueError:
                        return self._json({"ok": False, "mensagem": "Parâmetros inválidos."}, 400)
                    return self._json(serie(centro, dias))
                return self._json({"ok": False, "mensagem": "Não encontrado."}, 404)
            except Exception as e:  # nunca derruba o servidor por causa de um pedido
                return self._json({"ok": False, "mensagem": f"Erro interno: {e}"}, 500)

        def log_message(self, *args) -> None:  # silencia o log de cada requisição
            pass

    servidor = None
    for p in range(porta, porta + 11):  # tenta a porta pedida e as 10 seguintes
        try:
            servidor = ThreadingHTTPServer(("127.0.0.1", p), Handler)
            break
        except OSError:
            continue
    if servidor is None:
        raise SystemExit(f"Não consegui abrir nenhuma porta entre {porta} e {porta + 10}.")
    url = f"http://127.0.0.1:{servidor.server_address[1]}"
    print(f"\nPronto! Abra no navegador: {url}\n(Para encerrar, feche esta janela ou aperte Ctrl+C.)")
    if abrir:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        print("\nEncerrado.")
    finally:
        servidor.server_close()
