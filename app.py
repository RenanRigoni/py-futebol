import os
import requests
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# liberar acesso do frontend (Firebase depois)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# CONFIG
API_KEY = os.getenv("API_KEY")
SPORT = "soccer_brazil_campeonato"

SIMULATIONS = 300_000  # padrão leve para produção

DRAW_BOOST = 1.15
EQUILIBRIUM_DRAW_BOOST = 1.05
HOME_ADVANTAGE = 0.25


# =========================
# FUNÇÕES BASE
# =========================

def get_games():
    url = f"https://api.the-odds-api.com/v4/sports/{SPORT}/odds/?apiKey={API_KEY}&regions=eu&markets=h2h"

    resp = requests.get(url, timeout=30)
    data = resp.json()

    if isinstance(data, dict):
        raise Exception("Erro na API de odds")

    games = []

    for game in data:
        home = game["home_team"]
        away = game["away_team"]

        home_odds, draw_odds, away_odds = [], [], []

        for book in game.get("bookmakers", []):
            for market in book.get("markets", []):
                if market.get("key") != "h2h":
                    continue

                for outcome in market.get("outcomes", []):
                    if outcome["name"] == home:
                        home_odds.append(outcome["price"])
                    elif outcome["name"] == away:
                        away_odds.append(outcome["price"])
                    elif outcome["name"] == "Draw":
                        draw_odds.append(outcome["price"])

        if not home_odds or not draw_odds or not away_odds:
            continue

        home_odd = np.mean(home_odds)
        draw_odd = np.mean(draw_odds)
        away_odd = np.mean(away_odds)

        p1 = 1 / home_odd
        px = 1 / draw_odd
        p2 = 1 / away_odd

        total = p1 + px + p2
        p1 /= total
        px /= total
        p2 /= total

        px *= DRAW_BOOST

        if abs(home_odd - away_odd) < 0.40:
            px *= EQUILIBRIUM_DRAW_BOOST

        total = p1 + px + p2
        p1 /= total
        px /= total
        p2 /= total

        total_goals = 2.6
        home_xg = total_goals * (p1 + px / 2) + HOME_ADVANTAGE
        away_xg = total_goals * (p2 + px / 2)

        games.append((home, away, home_xg, away_xg))

    return games[:10]


def simulate(games):
    N = len(games)

    home_xg = np.array([g[2] for g in games])
    away_xg = np.array([g[3] for g in games])

    home_goals = np.random.poisson(home_xg[:, None], (N, SIMULATIONS))
    away_goals = np.random.poisson(away_xg[:, None], (N, SIMULATIONS))

    mask_00 = (home_goals == 0) & (away_goals == 0)
    home_goals[mask_00] += np.random.binomial(1, 0.25, mask_00.sum())

    mask_11 = (home_goals == 1) & (away_goals == 1)
    away_goals[mask_11] += np.random.binomial(1, 0.20, mask_11.sum())

    results = np.full((N, SIMULATIONS), 1, dtype=np.int8)
    results[home_goals < away_goals] = 2
    results[home_goals == away_goals] = 0

    return results


def get_probabilities(results):
    probs = []

    for res in results:
        p1 = np.mean(res == 1)
        px = np.mean(res == 0)
        p2 = np.mean(res == 2)

        probs.append({
            "1": p1,
            "X": px,
            "2": p2,
            "1X": p1 + px,
            "X2": px + p2,
            "12": p1 + p2
        })

    return probs


# =========================
# GERAÇÃO DE BILHETE
# =========================

def generate_ticket(games, probs, mode="misto"):
    ticket = []

    for i in range(len(games)):
        p = probs[i]

        if mode == "seco":
            best = max(["1", "X", "2"], key=lambda x: p[x])

        elif mode == "protecao":
            best = max(["1X", "X2", "12"], key=lambda x: p[x])

        else:  # misto
            best_single = max(["1", "X", "2"], key=lambda x: p[x])
            best_double = max(["1X", "X2", "12"], key=lambda x: p[x])

            if p[best_double] - p[best_single] > 0.12:
                best = best_double
            else:
                best = best_single

        ticket.append((best, p[best]))

    return ticket


def format_response(games, ticket):
    output = []

    for i, (pick, prob) in enumerate(ticket):
        home, away = games[i][0], games[i][1]

        if pick == "1":
            text = home
        elif pick == "2":
            text = away
        elif pick == "X":
            text = "Empate"
        elif pick == "1X":
            text = f"{home} ou Empate"
        elif pick == "X2":
            text = f"{away} ou Empate"
        else:
            text = f"{home} ou {away}"

        output.append({
            "jogo": f"{home} x {away}",
            "palpite": text,
            "confianca": round(prob * 100, 1)
        })

    return output


# =========================
# ENDPOINT
# =========================

@app.get("/")
def home():
    return {"status": "API rodando"}


@app.post("/generate")
def generate(qtd: int = 1, modo: str = "misto"):
    games = get_games()

    if len(games) < 5:
        return {"erro": "Poucos jogos disponíveis"}

    results = simulate(games)
    probs = get_probabilities(results)

    bilhetes = []

    for _ in range(qtd):
        ticket = generate_ticket(games, probs, modo)
        bilhetes.append(format_response(games, ticket))

    return {
        "bilhetes": bilhetes
    }