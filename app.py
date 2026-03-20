import os
import math
import heapq
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
MAX_DOUBLE_CHANCES_PROTECAO = 6


class FixacaoInput(BaseModel):
    jogo: str
    escolha: str


class GenerateRequest(BaseModel):
    fixacoes: list[FixacaoInput] = Field(default_factory=list)


def validate_env():
    if not API_KEY:
        raise RuntimeError("API_KEY não configurada")


def normalize_probs(p1: float, px: float, p2: float):
    total = p1 + px + p2
    if total <= 0:
        raise RuntimeError("Probabilidades inválidas")
    return p1 / total, px / total, p2 / total


def build_game_label(home: str, away: str) -> str:
    return f"{home} x {away}"


def filter_double_options(double_probs: dict[str, float], tipo_dupla: str) -> dict[str, float]:
    if tipo_dupla == "evitar_12":
        filtered = {k: v for k, v in double_probs.items() if k in {"1X", "X2"}}
        return filtered if filtered else double_probs
    return double_probs


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

        ranked_singles = sorted(single.items(), key=lambda x: x[1], reverse=True)
        ranked_doubles = sorted(double.items(), key=lambda x: x[1], reverse=True)

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

        infos.append(
            {
                "single_probs": single,
                "double_probs": double,
                "best_single": best_single,
                "best_double": best_double,
                "ranked_singles": ranked_singles,
                "ranked_doubles": ranked_doubles,
                "chosen_type": chosen_type,
                "protection_score": best_double[1] - best_single[1],
            }
        )

    return infos


def normalize_fixed_choice(escolha, home, away):
    e = escolha.strip().lower()
    if e in [home.lower(), "1"]:
        return "1"
    if e in [away.lower(), "2"]:
        return "2"
    if e in ["empate", "x"]:
        return "X"
    return None


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


def get_allowed_double_games(infos, modo):
    if modo == "misto":
        limit = MAX_DOUBLE_CHANCES_MISTO
    elif modo == "protecao":
        limit = MAX_DOUBLE_CHANCES_PROTECAO
    else:
        return set()

    ranked = sorted(
        enumerate(infos),
        key=lambda x: x[1]["protection_score"],
        reverse=True,
    )

    return {idx for idx, _ in ranked[:limit]}


def unique_candidates(candidates):
    seen = set()
    out = []
    for opt, prob in candidates:
        if opt not in seen:
            seen.add(opt)
            out.append((opt, float(prob)))
    return out


def build_candidate_lists(games, infos, modo, fix_map):
    allowed_double_games = get_allowed_double_games(infos, modo)
    candidate_lists = []
    fixacoes_aplicadas = []

    for idx, game in enumerate(games):
        label = game["label"]
        info = infos[idx]

        # fixação manual sempre tem prioridade
        if label in fix_map:
            forced = normalize_fixed_choice(fix_map[label], game["home"], game["away"])
            if forced is not None:
                prob = info["single_probs"][forced]
                candidate_lists.append([(forced, prob)])
                fixacoes_aplicadas.append(
                    {
                        "jogo": label,
                        "escolha_original": fix_map[label],
                        "mercado_forcado": forced,
                    }
                )
                continue

        if modo == "seco":
            # sempre 3 possibilidades por jogo
            candidates = info["ranked_singles"][:3]
            candidate_lists.append(unique_candidates(candidates))
            continue

        if modo == "misto":
            if idx in allowed_double_games and info["chosen_type"] == "double":
                candidates = [
                    *info["ranked_doubles"][:3],
                    *info["ranked_singles"][:3],
                ]
            else:
                candidates = [
                    *info["ranked_singles"][:3],
                    *info["ranked_doubles"][:2],
                ]
            candidate_lists.append(unique_candidates(candidates))
            continue

        if modo == "protecao":
            if idx in allowed_double_games:
                candidates = [
                    *info["ranked_doubles"][:3],
                    *info["ranked_singles"][:3],
                ]
            else:
                candidates = [
                    *info["ranked_singles"][:3],
                    *info["ranked_doubles"][:2],
                ]
            candidate_lists.append(unique_candidates(candidates))
            continue

        candidate_lists.append(unique_candidates(info["ranked_singles"][:3]))

    return candidate_lists, fixacoes_aplicadas


def combo_score(candidate_lists, idxs):
    score = 0.0
    for game_idx, cand_idx in enumerate(idxs):
        _opt, prob = candidate_lists[game_idx][cand_idx]
        score += math.log(max(prob, 1e-9))
    return score


def combo_signature(candidate_lists, idxs):
    return tuple(candidate_lists[i][idxs[i]][0] for i in range(len(idxs)))


def combo_ticket(candidate_lists, idxs):
    return [candidate_lists[i][idxs[i]] for i in range(len(idxs))]


def generate_unique_tickets(candidate_lists, qtd):
    base = tuple(0 for _ in candidate_lists)
    heap = [(-combo_score(candidate_lists, base), base)]
    seen_states = {base}
    seen_signatures = set()
    tickets = []

    while heap and len(tickets) < qtd:
        neg_score, state = heapq.heappop(heap)
        signature = combo_signature(candidate_lists, state)

        if signature not in seen_signatures:
            seen_signatures.add(signature)
            tickets.append(combo_ticket(candidate_lists, state))

        for i in range(len(state)):
            if state[i] + 1 < len(candidate_lists[i]):
                nxt = list(state)
                nxt[i] += 1
                nxt = tuple(nxt)

                if nxt not in seen_states:
                    seen_states.add(nxt)
                    heapq.heappush(
                        heap,
                        (-combo_score(candidate_lists, nxt), nxt),
                    )

    return tickets


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

    modo = modo.lower().strip()
    if modo not in {"seco", "misto", "protecao"}:
        return {"erro": "Modo inválido. Use: seco, misto ou protecao."}

    tipo_dupla = tipo_dupla.lower().strip()
    if tipo_dupla not in {"padrao", "evitar_12"}:
        return {"erro": "tipo_dupla inválido. Use: padrao ou evitar_12."}

    results = simulate_results(games)
    infos = build_game_infos(results, tipo_dupla)

    fixacoes = payload.fixacoes if payload else []
    valid_labels = {g["label"] for g in games}
    fix_map = {}

    for f in fixacoes:
        if f.jogo in valid_labels:
            fix_map[f.jogo] = f.escolha

    candidate_lists, fixacoes_aplicadas = build_candidate_lists(games, infos, modo, fix_map)
    tickets = generate_unique_tickets(candidate_lists, qtd)

    if len(tickets) < qtd:
        return {
            "erro": (
                f"Não foi possível gerar {qtd} bilhetes únicos com as restrições atuais. "
                f"Foram possíveis apenas {len(tickets)}."
            )
        }

    bilhetes = []
    for ticket in tickets:
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
