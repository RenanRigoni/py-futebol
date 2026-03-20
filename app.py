import os
import requests
import numpy as np
from typing import Optional
from fastapi import FastAPI, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="py-futebol API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.getenv("API_KEY")
SPORT = "soccer_brazil_campeonato"

SIMULATIONS = 300_000

DRAW_BOOST = 1.15
EQUILIBRIUM_DRAW_BOOST = 1.05
HOME_ADVANTAGE = 0.25

MAX_GAMES = 10

STRONG_SINGLE_THRESHOLD = 0.58
VERY_STRONG_SINGLE_THRESHOLD = 0.67
DOUBLE_ADVANTAGE_MARGIN = 0.12


class FixacaoInput(BaseModel):
    jogo: str
    escolha: str


class GenerateRequest(BaseModel):
    fixacoes: list[FixacaoInput] = Field(default_factory=list)


def filter_double_options(double_probs, tipo_dupla: str):
    if tipo_dupla == "evitar_12":
        return {k: v for k, v in double_probs.items() if k in {"1X", "X2"}}
    return double_probs


def validate_env():
    if not API_KEY:
        raise RuntimeError("API_KEY não configurada")


def normalize_probs(p1, px, p2):
    total = p1 + px + p2
    if total <= 0:
        raise RuntimeError("Probabilidades inválidas")
    return p1 / total, px / total, p2 / total


def build_game_label(home, away):
    return f"{home} x {away}"


def get_games():
    validate_env()

    url = (
        f"https://api.the-odds-api.com/v4/sports/{SPORT}/odds/"
        f"?apiKey={API_KEY}&regions=eu&markets=h2h"
    )

    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if isinstance(data, dict):
        message = data.get("message") or data.get("error") or "Erro na API de odds"
        raise RuntimeError(message)

    games = []

    for game in data:
        home = game.get("home_team")
        away = game.get("away_team")
        if not home or not away:
            continue

        home_odds, draw_odds, away_odds = [], [], []

        for book in game.get("bookmakers", []):
            for market in book.get("markets", []):
                if market.get("key") != "h2h":
                    continue

                for o in market.get("outcomes", []):
                    name = o.get("name")
                    price = o.get("price")

                    if not isinstance(price, (int, float)) or price <= 1:
                        continue

                    if name == home:
                        home_odds.append(price)
                    elif name == away:
                        away_odds.append(price)
                    elif name == "Draw":
                        draw_odds.append(price)

        if not home_odds or not draw_odds or not away_odds:
            continue

        home_odd = float(np.mean(home_odds))
        draw_odd = float(np.mean(draw_odds))
        away_odd = float(np.mean(away_odds))

        p1, px, p2 = normalize_probs(1 / home_odd, 1 / draw_odd, 1 / away_odd)

        px *= DRAW_BOOST
        if abs(home_odd - away_odd) < 0.4:
            px *= EQUILIBRIUM_DRAW_BOOST

        p1, px, p2 = normalize_probs(p1, px, p2)

        total_goals = 2.6
        home_xg = total_goals * (p1 + px / 2) + HOME_ADVANTAGE
        away_xg = total_goals * (p2 + px / 2)

        games.append(
            {
                "home": home,
                "away": away,
                "label": build_game_label(home, away),
                "home_xg": float(home_xg),
                "away_xg": float(away_xg),
            }
        )

    return games[:MAX_GAMES]


def simulate_results(games):
    n = len(games)

    home_xg = np.array([g["home_xg"] for g in games], dtype=float)
    away_xg = np.array([g["away_xg"] for g in games], dtype=float)

    home = np.random.poisson(home_xg[:, None], (n, SIMULATIONS))
    away = np.random.poisson(away_xg[:, None], (n, SIMULATIONS))

    results = np.full((n, SIMULATIONS), 1, dtype=np.int8)
    results[home < away] = 2
    results[home == away] = 0

    return results


def option_probability(arr, opt):
    if opt == "1":
        return float(np.mean(arr == 1))
    if opt == "X":
        return float(np.mean(arr == 0))
    if opt == "2":
        return float(np.mean(arr == 2))
    if opt == "1X":
        return float(np.mean((arr == 1) | (arr == 0)))
    if opt == "X2":
        return float(np.mean((arr == 0) | (arr == 2)))
    if opt == "12":
        return float(np.mean((arr == 1) | (arr == 2)))
    raise ValueError(f"Opção inválida: {opt}")


def build_game_infos(results, tipo_dupla="padrao"):
    infos = []

    for i in range(results.shape[0]):
        arr = results[i]

        single = {k: option_probability(arr, k) for k in ["1", "X", "2"]}
        double_raw = {k: option_probability(arr, k) for k in ["1X", "X2", "12"]}

        double = filter_double_options(double_raw, tipo_dupla)
        if not double:
            double = double_raw

        best_single = max(single.items(), key=lambda x: x[1])
        best_double = max(double.items(), key=lambda x: x[1])

        infos.append(
            {
                "single_probs": single,
                "double_probs": double,
                "best_single": best_single,
                "best_double": best_double,
                "protection_score": best_double[1] - best_single[1],
            }
        )

    return infos


def generate_ticket_from_infos(infos, modo):
    modo = modo.lower().strip()
    ticket = []

    for info in infos:
        if modo == "seco":
            ticket.append(info["best_single"])
        elif modo in {"misto", "protecao"}:
            if info["protection_score"] > DOUBLE_ADVANTAGE_MARGIN:
                ticket.append(info["best_double"])
            else:
                ticket.append(info["best_single"])
        else:
            ticket.append(info["best_single"])

    return ticket


def normalize_fixed_choice(escolha, home, away):
    e = escolha.strip().lower()
    if e in [home.lower(), "1"]:
        return "1"
    if e in [away.lower(), "2"]:
        return "2"
    if e in ["empate", "x"]:
        return "X"
    return None


def apply_fixacoes(ticket, games, infos, fix_map):
    out = []

    for i, (opt, prob) in enumerate(ticket):
        label = games[i]["label"]

        if label in fix_map:
            forced = normalize_fixed_choice(fix_map[label], games[i]["home"], games[i]["away"])
            if forced is not None:
                prob = infos[i]["single_probs"][forced]
                out.append((forced, prob))
                continue

        out.append((opt, prob))

    return out


def symbol_to_text(opt, home, away):
    if opt == "1":
        return home
    if opt == "2":
        return away
    if opt == "X":
        return "Empate"
    if opt == "1X":
        return f"{home} ou Empate"
    if opt == "X2":
        return f"{away} ou Empate"
    if opt == "12":
        return f"{home} ou {away}"
    return opt


def count_double_markets(ticket):
    return sum(1 for opt, _ in ticket if opt in {"1X", "X2", "12"})


def format_ticket(games, ticket):
    out = []

    for i, (opt, prob) in enumerate(ticket):
        home = games[i]["home"]
        away = games[i]["away"]

        out.append(
            {
                "jogo": games[i]["label"],
                "palpite": symbol_to_text(opt, home, away),
                "mercado": opt,
                "confianca": round(prob * 100, 1),
            }
        )

    return out


@app.get("/")
def home():
    return {"status": "API rodando"}


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/games")
def list_games():
    try:
        games = get_games()
    except Exception as e:
        return {"erro": f"Falha ao carregar jogos: {str(e)}"}

    return {
        "qtd_jogos": len(games),
        "jogos": [g["label"] for g in games],
        "detalhes": games,
    }


@app.post("/generate")
def generate(
    payload: Optional[GenerateRequest] = Body(default=None),
    qtd: int = Query(1, ge=1, le=10),
    modo: str = Query("misto"),
    tipo_dupla: str = Query("padrao"),
):
    try:
        games = get_games()
    except Exception as e:
        return {"erro": f"Falha ao obter odds: {str(e)}"}

    if not games:
        return {"erro": "Nenhum jogo encontrado."}

    results = simulate_results(games)
    infos = build_game_infos(results, tipo_dupla)

    fixacoes = payload.fixacoes if payload else []
    fix_map = {}
    fixacoes_aplicadas = []

    valid_labels = {g["label"] for g in games}

    for f in fixacoes:
        if f.jogo in valid_labels:
            fix_map[f.jogo] = f.escolha
            fixacoes_aplicadas.append(
                {
                    "jogo": f.jogo,
                    "escolha_original": f.escolha,
                }
            )

    bilhetes = []

    for _ in range(qtd):
        ticket = generate_ticket_from_infos(infos, modo)
        ticket = apply_fixacoes(ticket, games, infos, fix_map)

        bilhetes.append(
            {
                "modo": modo,
                "tipo_dupla": tipo_dupla,
                "duplas_usadas": count_double_markets(ticket),
                "fixacoes_aplicadas": fixacoes_aplicadas,
                "jogos": format_ticket(games, ticket),
            }
        )

    return {
        "qtd_solicitada": qtd,
        "qtd_gerada": len(bilhetes),
        "modo": modo,
        "tipo_dupla": tipo_dupla,
        "simulacoes": SIMULATIONS,
        "fixacoes_recebidas": len(fixacoes),
        "fixacoes_validas": len(fixacoes_aplicadas),
        "bilhetes": bilhetes,
    }
