import json
import math
import statistics
from collections import defaultdict
from scipy.stats import norm, poisson


with open("projections.json") as f:
    projections = json.load(f)

with open("prop_trends.json") as f:
    trends = json.load(f)


ALLOWED_PROPS = {
    "points",
    "pointsRebounds",
    "pointsAssists",
    "reboundsAssists",
    "pointsReboundsAssists",
}


proj_map = {p["id"]: p for p in projections}
trend_map = {p["id"]: p for p in trends}


class MathTools:

    def american_to_prob(self, odds):
        if odds > 0:
            return 100 / (odds + 100)
        return -odds / (-odds + 100)

    def remove_vig(self, over_odds, under_odds):
        p_over = self.american_to_prob(over_odds)
        p_under = self.american_to_prob(under_odds)
        total = p_over + p_under
        return p_over / total

    def expected_value(self, prob, odds):
        if odds > 0:
            return prob * (odds / 100) - (1 - prob)
        return prob - (1 - prob) * (100 / -odds)

    def kelly(self, prob, odds):
        if odds > 0:
            b = odds / 100
        else:
            b = 100 / -odds
        f = (prob * (b + 1) - 1) / b
        return max(f, 0)

    def poisson_over(self, mean, line):
        k = math.floor(line)
        return 1 - poisson.cdf(k, mean)

    def normal_over(self, mean, line, std):
        z = (mean - line) / std
        return norm.cdf(z)

    def logistic(self, x):
        return 1 / (1 + math.exp(-x))


math_tools = MathTools()


class ModelEngine:

    def estimate_std(self, mean, l10, l20):
        vol = abs(l10 - l20) / 100
        base = max(1.5, mean * 0.16)
        return base * (1 + vol)

    def projection_elasticity(self, mean, line):
        if line == 0:
            return 0
        return (mean - line) / line

    def mean_reversion(self, rate, l10):
        return 1 - abs(rate - l10) * 0.5

    def regime_adjustment(self, l5, l10):
        delta = l5 - l10
        return 1 + delta * 0.4

    def volatility_penalty(self, l10, l20):
        return 1 - abs(l10 - l20) * 0.4

    def consistency(self, rate, l10, l20):
        return 1 - (abs(rate - l10) + abs(l10 - l20)) * 0.3


model_engine = ModelEngine()


class AdaptiveBlend:

    def blend(self, model_p, trend_p, market_p):
        disagreement = abs(model_p - market_p)

        model_weight = 0.35 + disagreement * 0.6
        trend_weight = 0.35
        market_weight = 1 - model_weight - trend_weight

        raw = (
            model_weight * model_p +
            trend_weight * trend_p +
            market_weight * market_p
        )

        return max(min(raw, 0.99), 0.01)

    def shrink_to_market(self, final_p, market_p, confidence):
        shrink = (1 - confidence) * 0.5
        return final_p * (1 - shrink) + market_p * shrink

    def calibration(self, p):
        return math_tools.logistic((p - 0.5) * 6)


adaptive = AdaptiveBlend()


player_best = {}
raw_scores = []

for pid, proj in proj_map.items():

    if pid not in trend_map:
        continue

    name = proj["name"]
    team = proj["team"]

    player_trend = trend_map[pid]
    player_proj = proj["projections"]

    best = None

    for prop, data in player_trend.items():

        if prop not in ALLOWED_PROPS:
            continue

        if prop not in player_proj:
            continue

        line = data.get("line")
        over_odds = data.get("over")
        under_odds = data.get("under")

        if not line or not over_odds or not under_odds:
            continue

        mean = player_proj[prop]

        market_p = math_tools.remove_vig(over_odds, under_odds)

        rate = data.get("rate", 50) / 100
        l5 = data.get("l5Rate", 50) / 100
        l10 = data.get("l10Rate", 50) / 100
        l20 = data.get("l20Rate", 50) / 100
        opp = data.get("oppDef", 15)

        std = model_engine.estimate_std(
            mean,
            data.get("l10Rate", 50),
            data.get("l20Rate", 50),
        )

        if prop == "points" and mean < 25:
            model_p = math_tools.poisson_over(mean, line)
        else:
            model_p = math_tools.normal_over(mean, line, std)

        elasticity = model_engine.projection_elasticity(mean, line)
        trend_p = (
            0.45 * rate +
            0.25 * l5 +
            0.2 * l10 +
            0.1 * l20
        )

        trend_p *= (1 + elasticity)
        trend_p *= model_engine.regime_adjustment(l5, l10)

        blended = adaptive.blend(model_p, trend_p, market_p)

        confidence = 1 - abs(model_p - trend_p)

        blended = adaptive.shrink_to_market(blended, market_p, confidence)

        calibrated = adaptive.calibration(blended)

        ev = math_tools.expected_value(calibrated, over_odds)
        kelly = math_tools.kelly(calibrated, over_odds)

        score = (
            ev *
            confidence *
            model_engine.volatility_penalty(l10, l20) *
            model_engine.mean_reversion(rate, l10) *
            model_engine.consistency(rate, l10, l20)
        )

        candidate = {
            "player": name,
            "team": team,
            "prop": prop,
            "line": line,
            "projection": round(mean, 2),
            "prob": round(calibrated, 3),
            "ev": round(ev, 3),
            "kelly": round(kelly, 3),
            "score": score,
        }

        raw_scores.append(score)

        if best is None or candidate["score"] > best["score"]:
            best = candidate

    if best:
        player_best[name] = best


score_values = list(raw_scores)
mean_score = statistics.mean(score_values)
std_score = statistics.stdev(score_values) if len(score_values) > 1 else 1


ranked = []

for p in player_best.values():
    normalized = (p["score"] - mean_score) / std_score
    p["norm_score"] = normalized
    ranked.append(p)


ranked.sort(key=lambda x: x["norm_score"], reverse=True)


print("\nTop 3 Props\n")

for i, r in enumerate(ranked[:3], 1):
    print(
        f"{i}. {r['player']} ({r['team']}) "
        f"{r['prop']} over {r['line']} | "
        f"Proj: {r['projection']} | "
        f"Prob: {r['prob']} | "
        f"EV: {round(r['ev'],3)} | "
        f"Kelly: {r['kelly']}"
    )
