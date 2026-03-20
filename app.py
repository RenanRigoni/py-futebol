aqui meu .py^:

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
MAX_DOUBLE_CHANCES_MISTO = 3
MAX_DOUBLE_CHANCES_PROTECAO = 5


class FixacaoInput(BaseModel):
    jogo: str
    escolha: str


class GenerateRequest(BaseModel):
    fixacoes: list[FixacaoInput] = Field(default_factory=list)


# 🔥 NOVO FILTRO
def filter_double_options(double_probs, tipo_dupla: str):
    if tipo_dupla == "evitar_12":
        return {k: v for k, v in double_probs.items() if k in {"1X", "X2"}}
    return double_probs


def validate_env():
    if not API_KEY:
        raise RuntimeError("API_KEY não configurada")


def normalize_probs(p1, px, p2):
    total = p1 + px + p2
    return p1 / total, px / total, p2 / total


def build_game_label(home, away):
    return f"{home} x {away}"


def get_games():
    validate_env()

    url = f"https://api.the-odds-api.com/v4/sports/{SPORT}/odds/?apiKey={API_KEY}&regions=eu&markets=h2h"
    data = requests.get(url).json()

    games = []

    for game in data:
        home = game["home_team"]
        away = game["away_team"]

        home_odds, draw_odds, away_odds = [], [], []

        for book in game["bookmakers"]:
            for market in book["markets"]:
                if market["key"] != "h2h":
                    continue
                for o in market["outcomes"]:
                    if o["name"] == home:
                        home_odds.append(o["price"])
                    elif o["name"] == away:
                        away_odds.append(o["price"])
                    elif o["name"] == "Draw":
                        draw_odds.append(o["price"])

        if not home_odds or not draw_odds or not away_odds:
            continue

        home_odd = np.mean(home_odds)
        draw_odd = np.mean(draw_odds)
        away_odd = np.mean(away_odds)

        p1, px, p2 = normalize_probs(1/home_odd, 1/draw_odd, 1/away_odd)

        px *= DRAW_BOOST
        if abs(home_odd - away_odd) < 0.4:
            px *= EQUILIBRIUM_DRAW_BOOST

        p1, px, p2 = normalize_probs(p1, px, p2)

        total = 2.6
        home_xg = total * (p1 + px/2) + HOME_ADVANTAGE
        away_xg = total * (p2 + px/2)

        games.append({
            "home": home,
            "away": away,
            "label": build_game_label(home, away),
            "home_xg": home_xg,
            "away_xg": away_xg
        })

    return games[:MAX_GAMES]


def simulate_results(games):
    n = len(games)

    home_xg = np.array([g["home_xg"] for g in games])
    away_xg = np.array([g["away_xg"] for g in games])

    home = np.random.poisson(home_xg[:, None], (n, SIMULATIONS))
    away = np.random.poisson(away_xg[:, None], (n, SIMULATIONS))

    results = np.full((n, SIMULATIONS), 1)
    results[home < away] = 2
    results[home == away] = 0

    return results


def option_probability(arr, opt):
    if opt == "1": return np.mean(arr == 1)
    if opt == "X": return np.mean(arr == 0)
    if opt == "2": return np.mean(arr == 2)
    if opt == "1X": return np.mean((arr == 1)|(arr == 0))
    if opt == "X2": return np.mean((arr == 0)|(arr == 2))
    if opt == "12": return np.mean((arr == 1)|(arr == 2))


def build_game_infos(results, tipo_dupla="padrao"):
    infos = []

    for i in range(results.shape[0]):
        arr = results[i]

        single = {k: option_probability(arr, k) for k in ["1","X","2"]}
        double_raw = {k: option_probability(arr, k) for k in ["1X","X2","12"]}

        double = filter_double_options(double_raw, tipo_dupla)

        best_single = max(single.items(), key=lambda x: x[1])
        best_double = max(double.items(), key=lambda x: x[1])

        infos.append({
            "single_probs": single,
            "double_probs": double,
            "best_single": best_single,
            "best_double": best_double,
            "protection_score": best_double[1] - best_single[1]
        })

    return infos


def generate_ticket_from_infos(infos, modo):
    ticket = []

    for info in infos:
        if modo == "seco":
            ticket.append(info["best_single"])
        else:
            if info["protection_score"] > DOUBLE_ADVANTAGE_MARGIN:
                ticket.append(info["best_double"])
            else:
                ticket.append(info["best_single"])

    return ticket


def normalize_fixed_choice(escolha, home, away):
    e = escolha.lower()
    if e in [home.lower(),"1"]: return "1"
    if e in [away.lower(),"2"]: return "2"
    if e in ["empate","x"]: return "X"


def apply_fixacoes(ticket, games, infos, fix_map):
    out = []
    for i,(opt,prob) in enumerate(ticket):
        label = games[i]["label"]
        if label in fix_map:
            forced = normalize_fixed_choice(fix_map[label], games[i]["home"], games[i]["away"])
            prob = infos[i]["single_probs"][forced]
            out.append((forced, prob))
        else:
            out.append((opt, prob))
    return out


def format_ticket(games, ticket):
    out = []
    for i,(opt,prob) in enumerate(ticket):
        home = games[i]["home"]
        away = games[i]["away"]

        if opt=="1": txt=home
        elif opt=="2": txt=away
        elif opt=="X": txt="Empate"
        elif opt=="1X": txt=f"{home} ou Empate"
        elif opt=="X2": txt=f"{away} ou Empate"
        elif opt=="12": txt=f"{home} ou {away}"

        out.append({
            "jogo": games[i]["label"],
            "palpite": txt,
            "mercado": opt,
            "confianca": round(prob*100,1)
        })
    return out


@app.post("/generate")
def generate(
    payload: Optional[GenerateRequest] = Body(default=None),
    qtd: int = Query(1),
    modo: str = Query("misto"),
    tipo_dupla: str = Query("padrao")
):
    games = get_games()
    results = simulate_results(games)
    infos = build_game_infos(results, tipo_dupla)

    fix_map = {f.jogo:f.escolha for f in (payload.fixacoes if payload else [])}

    bilhetes = []

    for _ in range(qtd):
        t = generate_ticket_from_infos(infos, modo)
        t = apply_fixacoes(t, games, infos, fix_map)
        bilhetes.append({
            "modo": modo,
            "tipo_dupla": tipo_dupla,
            "jogos": format_ticket(games, t)
        })

    return {"bilhetes": bilhetes}
