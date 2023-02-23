from scripy.stats import binom


def chance_level(n, alpha = 0.001, p = 0.5):
    'n is the number of trials, alpha is the significance level, p is the chance level'
    k = binom.ppf(1-alpha, n, p)
    chance_level = k/n
    return chance_level
