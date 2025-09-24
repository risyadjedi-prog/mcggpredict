# mcggpredict
#!/usr/bin/env python3
"""
toolsmcggpredict.py
Predict next opponent in MagicChess (for Popol usage) using:
 - first-order Markov transitions from past rounds
 - heuristic scoring based on role, HP, win-streak
Usage:
    python toolsmcggpredict.py --history history.json --current enemy_5 --top 3
Or:
    python toolsmcggpredict.py --history history.json --top 5
History format (JSON list of rounds). Each round is a dict, example:
[
  {"order": ["enemy_1","enemy_4","enemy_2","enemy_5"], "meta": {
       "enemy_1": {"role":"assassin","hp":100,"win_streak":0},
       ...
  }},
  ...
]
If rounds are only sequences, meta can be omitted; heuristics will fallback to frequency.
"""
import argparse
import json
import math
from collections import defaultdict, Counter
from typing import List, Dict, Tuple

def load_history(path: str) -> List[Dict]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def build_markov(history: List[Dict]) -> Dict[str, Counter]:
    """
    Build first-order transition counts: from -> to
    history: list of rounds; each round has 'order' list (sequence of opponents)
    Return dict mapping from opponent -> Counter(next_opponent)
    """
    transitions = defaultdict(Counter)
    for rnd in history:
        order = rnd.get('order') or []
        for i in range(len(order)-1):
            a, b = order[i], order[i+1]
            transitions[a][b] += 1
    return transitions

def markov_predict(transitions: Dict[str, Counter], current: str, top_k: int=3) -> List[Tuple[str,float]]:
    """
    Given transitions and current opponent, return top_k next opponents with probability.
    If current not in transitions, fallback to global frequency.
    """
    if current in transitions and sum(transitions[current].values())>0:
        cnt = transitions[current]
    else:
        # fallback: aggregate all transitions
        aggregate = Counter()
        for c in transitions.values():
            aggregate.update(c)
        cnt = aggregate
    total = sum(cnt.values()) or 1
    probs = sorted([(op, cnt[op]/total) for op in cnt], key=lambda x: x[1], reverse=True)
    return probs[:top_k]

def overall_frequency(history: List[Dict], top_k: int=5) -> List[Tuple[str,float]]:
    freq = Counter()
    for rnd in history:
        for op in rnd.get('order', []):
            freq[op] += 1
    total = sum(freq.values()) or 1
    return [(op, freq[op]/total) for op,_ in freq.most_common(top_k)]

# Heuristic scoring for Popol trap placement
ROLE_PENALTIES = {
    # Higher score means more urgent to place trap in particular zone
    # We'll compute a score per opponent and also recommend trap placement
    "assassin": {"front": 0.2, "flank": 0.6, "back": 0.8},   # want trap near back/flank
    "mage":     {"front": 0.1, "flank": 0.3, "back": 0.9},
    "marksman": {"front": 0.1, "flank": 0.4, "back": 0.9},
    "fighter":  {"front": 0.6, "flank": 0.6, "back": 0.3},
    "tank":     {"front": 0.9, "flank": 0.5, "back": 0.2},
    "support":  {"front": 0.2, "flank": 0.5, "back": 0.8},
}

def heuristic_score(op_meta: Dict) -> Tuple[float, str]:
    """
    Compute heuristic score (0..1) and recommended trap placement ('front','flank','back')
    op_meta should contain 'role', 'hp', 'win_streak' (optional).
    """
    role = (op_meta.get('role') or 'unknown').lower()
    hp = float(op_meta.get('hp', 100))
    win = int(op_meta.get('win_streak', 0))

    # Normalize HP to 0..1 (assuming typical max ~ 1000 in some systems; but we scale adaptively)
    hp_score = 1.0 - (hp / (hp + 300.0))  # more HP -> lower urgency
    win_score = 1.0 - (1.0 / (1 + math.log(1+win))) if win>0 else 0.5  # winner tends to be cautious

    role_scores = ROLE_PENALTIES.get(role, {"front":0.5,"flank":0.5,"back":0.5})
    # compose an urgency score: role importance weighted + hp_score + win_score
    role_importance = max(role_scores.values())
    urgency = 0.45*role_importance + 0.35*hp_score + 0.20*win_score
    # pick placement with highest role preference * (1 + small hp modifier)
    placement = max(role_scores.items(), key=lambda kv: kv[1])[0]
    return (min(max(urgency, 0.0), 1.0), placement)

def combine_predictions(markov_list: List[Tuple[str,float]], heuristics: Dict[str,Dict], top_k:int=3) -> List[Dict]:
    """
    Combine markov probabilities with heuristic urgency to produce final ranked list.
    heuristics: mapping opponent -> meta dict (role/hp/win_streak)
    """
    results = []
    # convert markov list to dict
    total_prob = sum(p for _,p in markov_list) or 1.0
    for op, prob in markov_list:
        meta = heuristics.get(op, {})
        score, placement = heuristic_score(meta) if meta else (0.5, 'flank')
        # final_score: weighted combination (you can tweak)
        final = 0.65*prob + 0.35*score
        results.append({"opponent":op, "markov_prob":prob, "heuristic_urgency":score, "placement":placement, "final_score":final})
    # sort by final_score
    results.sort(key=lambda x: x['final_score'], reverse=True)
    return results[:top_k]

def collect_meta_from_history(history: List[Dict]) -> Dict[str,Dict]:
    """
    Build latest known meta per opponent (role/hp/win_streak) using the last occurrence in history.
    """
    meta = {}
    for rnd in history:
        m = rnd.get('meta', {})
        for op, info in m.items():
            meta[op] = info
    return meta

def main():
    p = argparse.ArgumentParser(description="Tool: mcgg predict - predict next opponent & Popol trap placement")
    p.add_argument('--history', required=True, help='JSON history file path')
    p.add_argument('--current', required=False, help='current opponent id (optional)')
    p.add_argument('--top', type=int, default=3, help='how many top predictions to show')
    args = p.parse_args()

    history = load_history(args.history)
    transitions = build_markov(history)
    meta_map = collect_meta_from_history(history)

    top_k = args.top
    if args.current:
        mlist = markov_predict(transitions, args.current, top_k=top_k*2)  # get some extras
    else:
        mlist = overall_frequency(history, top_k=top_k*2)

    combined = combine_predictions(mlist, meta_map, top_k=top_k)

    # Output nicely
    print("="*40)
    print("Tool: mcgg predict (Popol-focused)")
    print(f"History rounds: {len(history)}")
    if args.current:
        print(f"Current opponent: {args.current}")
    print("-"*40)
    for i, r in enumerate(combined, start=1):
        print(f"{i}. Opponent: {r['opponent']}")
        print(f"   Markov prob: {r['markov_prob']:.2%}  Heuristic urgency: {r['heuristic_urgency']:.2f}")
        print(f"   Recommended trap placement: {r['placement']}")
        print(f"   Final score: {r['final_score']:.3f}")
        print("-"*40)
    print("Notes: placement = 'front','flank','back'. Use this to pre-place Popol trap accordingly.")
    print("="*40)

if __name__ == '__main__':
    main()
