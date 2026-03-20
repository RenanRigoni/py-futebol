import os
import requests
import numpy as np
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

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


def validate_env() -> None:
    if not API_KEY:
        raise RuntimeError("API_KEY não configurada no ambiente.")


def normalize_probs(p1: float, px: float, p2: float) -> tuple[float, float, float]:
    total = p1 + px + p2
    if total <= 0:
        return 0.0, 0.0, 0.0
    return p1 / total, px / total, p2 / total


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

        home_odds = []
        draw_odds = []
        away_odds = []

        for book in game.get("bookmakers", []):
            for market in book.get("markets", []):
                if market.get("key") != "h2h":
                    continue

                for outcome in market.get("outcomes", []):
                    name = outcome.get("name")
                    price = outcome.get("price")

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

        p1 = 1 / home_odd
        px = 1 / draw_odd
        p2 = 1 / away_odd
        p1, px, p2 = normalize_probs(p1, px, p2)

        px *= DRAW_BOOST

        if abs(home_odd - away_odd) < 0.40:
            px *= EQUILIBRIUM_DRAW_BOOST

        p1, px, p2 = normalize_probs(p1, px, p2)

        total_goals = 2.6
        home_xg = total_goals * (p1 + px / 2) + HOME_ADVANTAGE
        away_xg = total_goals * (p2 + px / 2)

        games.append(
            {
                "home": home,
                "away": away,
                "home_xg": float(home_xg),
                "away_xg": float(away_xg),
            }
        )

    return games[:MAX_GAMES]


def simulate_results(games):
    n_games = len(games)

    home_xg = np.array([g["home_xg"] for g in games], dtype=float)
    away_xg = np.array([g["away_xg"] for g in games], dtype=float)

    home_goals = np.random.poisson(home_xg[:, None], (n_games, SIMULATIONS))
    away_goals = np.random.poisson(away_xg[:, None], (n_games, SIMULATIONS))

    mask_00 = (home_goals == 0) & (away_goals == 0)
    if mask_00.any():
        home_goals[mask_00] += np.random.binomial(1, 0.25, int(mask_00.sum()))

    mask_11 = (home_goals == 1) & (away_goals == 1)
    if mask_11.any():
        away_goals[mask_11] += np.random.binomial(1, 0.20, int(mask_11.sum()))

    results = np.full((n_games, SIMULATIONS), 1, dtype=np.int8)
    results[home_goals < away_goals] = 2
    results[home_goals == away_goals] = 0

    home_wins = np.sum(results == 1, axis=0)
    away_wins = np.sum(results == 2, axis=0)
    draws = np.sum(results == 0, axis=0)

    valid_mask = (draws <= 5) & (home_wins <= 7) & (away_wins <= 7)
    filtered = results[:, valid_mask]

    if filtered.shape[1] == 0:
        return results

    return filtered


def option_probability(res_array: np.ndarray, option: str) -> float:
    if option == "1":
        return float(np.mean(res_array == 1))
    if option == "X":
        return float(np.mean(res_array == 0))
    if option == "2":
        return float(np.mean(res_array == 2))
    if option == "1X":
        return float(np.mean((res_array == 1) | (res_array == 0)))
    if option == "X2":
        return float(np.mean((res_array == 0) | (res_array == 2)))
    if option == "12":
        return float(np.mean((res_array == 1) | (res_array == 2)))
    raise ValueError(f"Opção inválida: {option}")


def build_game_infos(results):
    single_options = ["1", "X", "2"]
    double_options = ["1X", "X2", "12"]

    game_infos = []

    for i in range(results.shape[0]):
        res_array = results[i]

        single_probs = {opt: option_probability(res_array, opt) for opt in single_options}
        double_probs = {opt: option_probability(res_array, opt) for opt in double_options}

        ranked_singles = sorted(single_probs.items(), key=lambda x: x[1], reverse=True)
        ranked_doubles = sorted(double_probs.items(), key=lambda x: x[1], reverse=True)

        best_single = ranked_singles[0]
        best_double = ranked_doubles[0]

        chosen_type = "single"
        if best_single[1] >= VERY_STRONG_SINGLE_THRESHOLD:
            chosen_type = "single"
        elif best_single[1] >= STRONG_SINGLE_THRESHOLD:
            chosen_type = "single"
        elif best_double[1] - best_single[1] >= DOUBLE_ADVANTAGE_MARGIN:
            chosen_type = "double"
        else:
            chosen_type = "single"

        protection_score = best_double[1] - best_single[1]

        game_infos.append(
            {
                "best_single": best_single,
                "best_double": best_double,
                "ranked_singles": ranked_singles,
                "ranked_doubles": ranked_doubles,
                "chosen_type": chosen_type,
                "protection_score": protection_score,
            }
        )

    return game_infos


def get_allowed_double_games(game_infos, max_double_chances: int):
    ranked = sorted(
        enumerate(game_infos),
        key=lambda x: x[1]["protection_score"],
        reverse=True,
    )
    return {idx for idx, _info in ranked[:max_double_chances]}


def choose_option_for_mode(info, mode: str, use_double: bool):
    if mode == "seco":
        return info["best_single"]

    if mode == "protecao":
        if use_double:
            return info["best_double"]
        return info["best_single"]

    if mode == "misto":
        if use_double:
            return info["best_double"]
        return info["best_single"]

    raise ValueError("Modo inválido.")


def generate_ticket_from_infos(game_infos, mode: str):
    ticket = []

    if mode == "seco":
        for info in game_infos:
            opt, prob = info["best_single"]
            ticket.append((opt, prob))
        return ticket

    if mode == "protecao":
        allowed_double_games = get_allowed_double_games(game_infos, MAX_DOUBLE_CHANCES_PROTECAO)

        for idx, info in enumerate(game_infos):
            use_double = idx in allowed_double_games
            opt, prob = choose_option_for_mode(info, mode, use_double)
            ticket.append((opt, prob))
        return ticket

    if mode == "misto":
        allowed_double_games = get_allowed_double_games(game_infos, MAX_DOUBLE_CHANCES_MISTO)

        for idx, info in enumerate(game_infos):
            use_double = idx in allowed_double_games and info["chosen_type"] == "double"
            opt, prob = choose_option_for_mode(info, mode, use_double)
            ticket.append((opt, prob))
        return ticket

    raise ValueError("Modo inválido. Use: seco, misto ou protecao.")


def diversify_ticket(base_ticket, game_infos, ticket_index: int, mode: str):
    if ticket_index == 0:
        return base_ticket

    diversified = []

    for idx, (current_opt, current_prob) in enumerate(base_ticket):
        info = game_infos[idx]

        if mode == "seco":
            candidates = info["ranked_singles"]
        elif current_opt in {"1X", "X2", "12"}:
            candidates = info["ranked_doubles"]
        else:
            candidates = info["ranked_singles"]

        if len(candidates) == 1:
            diversified.append((current_opt, current_prob))
            continue

        alt_idx = ticket_index % len(candidates)
        alt_opt, alt_prob = candidates[alt_idx]

        if current_opt in {"1X", "X2", "12"}:
            if alt_prob < 0.55:
                alt_opt, alt_prob = candidates[0]
        else:
            if alt_prob < 0.20:
                alt_opt, alt_prob = candidates[0]

        diversified.append((alt_opt, alt_prob))

    return diversified


def symbol_to_text(pick: str, home: str, away: str) -> str:
    if pick == "1":
        return home
    if pick == "2":
        return away
    if pick == "X":
        return "Empate"
    if pick == "1X":
        return f"{home} ou Empate"
    if pick == "X2":
        return f"{away} ou Empate"
    if pick == "12":
        return f"{home} ou {away}"
    return pick


def format_ticket_response(games, ticket, mode: str):
    output = []
    doubles_used = 0

    for i, (pick, prob) in enumerate(ticket):
        home = games[i]["home"]
        away = games[i]["away"]

        if pick in {"1X", "X2", "12"}:
            doubles_used += 1

        output.append(
            {
                "jogo": f"{home} x {away}",
                "palpite": symbol_to_text(pick, home, away),
                "mercado": pick,
                "confianca": round(prob * 100, 1),
            }
        )

    return {
        "modo": mode,
        "duplas_usadas": doubles_used,
        "jogos": output,
    }


@app.get("/")
def home():
    return {"status": "API rodando"}


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.post("/generate")
def generate(
    qtd: int = Query(1, ge=1, le=10),
    modo: str = Query("misto"),
):
    modo = modo.strip().lower()

    if modo not in {"seco", "misto", "protecao"}:
        return {"erro": "Modo inválido. Use: seco, misto ou protecao."}

    try:
        games = get_games()
    except Exception as e:
        return {"erro": f"Falha ao obter odds: {str(e)}"}

    if len(games) < 5:
        return {"erro": "Poucos jogos disponíveis para gerar bilhetes."}

    try:
        results = simulate_results(games)
        game_infos = build_game_infos(results)
    except Exception as e:
        return {"erro": f"Falha na simulação: {str(e)}"}

    bilhetes = []
    seen_signatures = set()

    base_ticket = generate_ticket_from_infos(game_infos, modo)

    for idx in range(qtd):
        ticket = diversify_ticket(base_ticket, game_infos, idx, modo)
        signature = tuple(opt for opt, _prob in ticket)

        if signature in seen_signatures:
            continue

        seen_signatures.add(signature)
        bilhetes.append(format_ticket_response(games, ticket, modo))

    if not bilhetes:
        bilhetes.append(format_ticket_response(games, base_ticket, modo))

    return {
        "qtd_solicitada": qtd,
        "qtd_gerada": len(bilhetes),
        "modo": modo,
        "simulacoes": SIMULATIONS,
        "bilhetes": bilhetes,
    }
