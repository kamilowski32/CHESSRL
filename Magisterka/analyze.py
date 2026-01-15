import pandas as pd
from pathlib import Path
from collections import defaultdict


# =========================
# POMOCNICZE
# =========================
def parse_result(row):
    if row["result"] == "1-0":
        return row["white"], row["black"], False
    elif row["result"] == "0-1":
        return row["black"], row["white"], False
    else:
        return None, None, True


def extract_agent(agent_name: str) -> str:
    """
    PPO_404_ep5000 -> PPO
    """
    return agent_name.split("_")[0]


def extract_episode(agent_name: str) -> int:
    """
    PPO_404_ep5000 -> 5000
    """
    return int(agent_name.split("_ep")[-1])


# =========================
# 1. AGENT — W/L/D
# =========================
def agent_wld(df: pd.DataFrame) -> pd.DataFrame:
    stats = defaultdict(lambda: {"Wins": 0, "Losses": 0, "Draws": 0})

    for _, row in df.iterrows():
        a_w = extract_agent(row["white"])
        a_b = extract_agent(row["black"])

        if(a_w == a_b):
            continue

        winner, loser, draw = parse_result(row)

        if draw:
            stats[a_w]["Draws"] += 1
            stats[a_b]["Draws"] += 1
        else:
            stats[extract_agent(winner)]["Wins"] += 1
            stats[extract_agent(loser)]["Losses"] += 1

    return (
        pd.DataFrame.from_dict(stats, orient="index")
        .reset_index()
        .rename(columns={"index": "Agent"})
        .sort_values("Wins", ascending=False)
    )


# =========================
# 2. AGENT + EPISODE — W/L/D
# =========================
def agent_episode_wld(df: pd.DataFrame) -> pd.DataFrame:
    stats = defaultdict(lambda: {"Wins": 0, "Losses": 0, "Draws": 0})

    for _, row in df.iterrows():
        w, b = row["white"], row["black"]

        aw, ab = extract_agent(w), extract_agent(b)
        ew, eb = extract_episode(w), extract_episode(b)

        if(aw == ab):
            continue

        winner, loser, draw = parse_result(row)

        if draw:
            stats[(aw, ew)]["Draws"] += 1
            stats[(ab, eb)]["Draws"] += 1
        else:
            stats[(extract_agent(winner), extract_episode(winner))]["Wins"] += 1
            stats[(extract_agent(loser), extract_episode(loser))]["Losses"] += 1

    return (
        pd.DataFrame.from_dict(stats, orient="index")
        .reset_index()
        .rename(columns={
            "level_0": "Agent",
            "level_1": "Episode"
        })
        .sort_values(["Episode", "Wins"], ascending=[True, False])
    )


# =========================
# 3. AGENT vs AGENT (H2H)
# =========================
def agent_head_to_head(df: pd.DataFrame) -> pd.DataFrame:
    stats = defaultdict(lambda: {"Wins1": 0, "Wins2": 0, "Draws": 0})

    for _, row in df.iterrows():
        a1 = extract_agent(row["white"])
        a2 = extract_agent(row["black"])

        if(a1 == a2):
            continue

        key = tuple(sorted([a1, a2]))
        winner, _, draw = parse_result(row)

        if draw:
            stats[key]["Draws"] += 1
        else:
            if extract_agent(winner) == key[0]:
                stats[key]["Wins1"] += 1
            else:
                stats[key]["Wins2"] += 1

    return pd.DataFrame([
        {
            "Agent1": k[0],
            "Agent2": k[1],
            **v
        }
        for k, v in stats.items()
    ]).sort_values("Wins1", ascending=False)


# =========================
# 4. AGENT1 EP1 vs AGENT2 EP2
# =========================
def agent_episode_head_to_head(df: pd.DataFrame) -> pd.DataFrame:
    stats = defaultdict(lambda: {"Wins1": 0, "Wins2": 0, "Draws": 0})

    for _, row in df.iterrows():
        a1, a2 = row["white"], row["black"]

        key = (
            extract_agent(a1),
            extract_episode(a1),
            extract_agent(a2),
            extract_episode(a2)
        )

        if(a1 == a2):
            continue

        winner, _, draw = parse_result(row)

        if draw:
            stats[key]["Draws"] += 1
        else:
            if winner == a1:
                stats[key]["Wins1"] += 1
            else:
                stats[key]["Wins2"] += 1

    return pd.DataFrame([
        {
            "Agent1": k[0],
            "Episode1": k[1],
            "Agent2": k[2],
            "Episode2": k[3],
            **v
        }
        for k, v in stats.items()
    ]).sort_values("Wins1", ascending=False)

# =========================
# 5. EPISODE vs EPISODE
# =========================
def episode_vs_episode(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ep1, Ep2, Wins1, Wins2, Draws
    """
    stats = defaultdict(lambda: {"Wins1": 0, "Wins2": 0, "Draws": 0})

    for _, row in df.iterrows():
        ep1 = extract_episode(row["white"])
        ep2 = extract_episode(row["black"])

        key = (ep1, ep2)

        winner, _, draw = parse_result(row)

        if draw:
            stats[key]["Draws"] += 1
        else:
            if winner == row["white"]:
                stats[key]["Wins1"] += 1
            else:
                stats[key]["Wins2"] += 1

    return (
        pd.DataFrame([
            {
                "Episode1": k[0],
                "Episode2": k[1],
                **v
            }
            for k, v in stats.items()
        ])
        .sort_values(["Episode1", "Episode2"])
    )



# =========================
# ANALIZA JEDNEGO TURNIEJU
# =========================
def analyze_tournament(results_csv: Path):
    df = pd.read_csv(results_csv)

    return {
        "agent": agent_wld(df),
        "agent_episode": agent_episode_wld(df),
        "agent_h2h": agent_head_to_head(df),
        "agent_episode_h2h": agent_episode_head_to_head(df),
        "episode_h2h": episode_vs_episode(df),
    }


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    TOURNAMENTS = [
        "pool_league_ep50000",
        "pool_league_ep100000",
        "pool_league_mixed_50000_vs_100000"
    ]

    BASE = Path("tournaments")

    for t in TOURNAMENTS:
        print("\n" + "=" * 80)
        print(f"TOURNAMENT: {t}")
        print("=" * 80)

        csv_path = BASE / t / "results.csv"
        stats = analyze_tournament(csv_path)

        print("\n[1] AGENT — W / L / D")
        print(stats["agent"].head())
        stats["agent"].to_csv(BASE / t / f"{t}_agent_wld.csv", index=False)

        print("\n[2] AGENT + EPISODE — W / L / D")
        print(stats["agent_episode"].head())
        stats["agent_episode"].to_csv(BASE / t / f"{t}_agent_episode_wld.csv", index=False)

        print("\n[3] AGENT vs AGENT")
        print(stats["agent_h2h"].head())
        stats["agent_h2h"].to_csv(BASE / t / f"{t}_agent_h2h.csv", index=False)

        print("\n[4] AGENT+EP vs AGENT+EP")
        print(stats["agent_episode_h2h"].head())
        stats["agent_episode_h2h"].to_csv(BASE / t / f"{t}_agent_episode_h2h.csv", index=False)

        print("\n[5] EPISODE vs EPISODE")
        print(stats["episode_h2h"].head())
        stats["episode_h2h"].to_csv(BASE / t / f"{t}_episode_vs_episode.csv", index=False)
