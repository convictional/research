def calculate_elo_change(winner_elo, loser_elo, k_factor=32):
    expected_score = 1 / (1 + 10 ** ((loser_elo - winner_elo) / 400))
    actual_score = 1
    elo_change = k_factor * (actual_score - expected_score)
    return elo_change


def update_elo_ratings(rankings, elo_ratings):
    for model, ranking in rankings.items():
        for other_model, other_ranking in rankings.items():
            if model != other_model:
                if ranking < other_ranking:
                    elo_change = calculate_elo_change(elo_ratings[model], elo_ratings[other_model])
                    elo_ratings[model] += elo_change
                    elo_ratings[other_model] -= elo_change
    return elo_ratings
