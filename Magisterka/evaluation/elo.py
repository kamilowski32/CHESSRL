import math

class EloRating:
    """
    Prosty system Elo.
    """

    def __init__(self, rating=1200, k_factor=32):
        self.rating = rating
        self.k = k_factor

    def expected_score(self, opponent_rating):
        return 1.0 / (1.0 + 10 ** ((opponent_rating - self.rating) / 400))

    def update(self, opponent_rating, score):
        """
        score:
            1.0 = win
            0.5 = draw
            0.0 = loss
        """
        expected = self.expected_score(opponent_rating)
        self.rating += self.k * (score - expected)
