"""Binomial valuation and probability calculations, independent of Streamlit."""
from dataclasses import dataclass
from math import exp, log, sqrt, erf, isfinite, comb
import numpy as np


@dataclass(frozen=True)
class Inputs:
    S: float = 100.0
    K: float = 100.0
    T: float = 1.0
    r: float = 0.08
    q: float = 0.0
    sigma: float = 0.30
    n: int = 3
    u: float | None = None
    d: float | None = None

    def terms(self):
        values = (self.S, self.K, self.T, self.r, self.q, self.sigma)
        if not all(isfinite(x) for x in values):
            raise ValueError('All inputs must be finite numbers.')
        if self.S <= 0 or self.K < 0 or self.T <= 0 or self.sigma < 0:
            raise ValueError('Stock price and maturity must be positive; strike and volatility cannot be negative.')
        if type(self.n) is not int or not 1 <= self.n <= 2000:
            raise ValueError('Choose a whole number of periods from 1 to 2,000.')
        dt = self.T / self.n
        if (self.u is None) != (self.d is None):
            raise ValueError('Supply both u and d, or calculate both from volatility.')
        u = exp(self.sigma * sqrt(dt)) if self.u is None else self.u
        d = 1 / u if self.d is None else self.d
        if not all(isfinite(x) for x in (u, d)) or not 0 < d < u:
            raise ValueError('The tree requires 0 < d < u. Use positive volatility or distinct factors.')
        growth = exp((self.r - self.q) * dt)
        if not d < growth < u:
            raise ValueError('No-arbitrage requires d < exp((r-q) times step length) < u. Adjust the factors, rates, volatility, or step count.')
        return dt, u, d, exp(self.r * dt), (growth - d) / (u - d)


def intrinsic(stock, K, kind):
    if kind not in ('Call', 'Put'):
        raise ValueError('Choose Call or Put.')
    return np.maximum(stock - K if kind == 'Call' else K - stock, 0.0)


def exercise_steps(n, style, dates=()):
    if style == 'European':
        return {n}
    if style == 'American':
        return set(range(n + 1))
    if style != 'Bermudan':
        raise ValueError('Unknown exercise style.')
    if any(type(i) is not int or i < 0 or i > n for i in dates):
        raise ValueError('Exercise dates must be integer time steps in the tree.')
    return set(dates) | {n}


def lattice(a, kind='Call', style='European', dates=(), barrier=None):
    dt, u, d, R, p = a.terms()
    allowed = exercise_steps(a.n, style, dates)
    if barrier is not None and (not isfinite(barrier) or barrier <= 0 or style != 'European'):
        raise ValueError('Use a positive barrier with European exercise.')
    stock = [a.S * u ** np.arange(i + 1) * d ** (i - np.arange(i + 1)) for i in range(a.n + 1)]
    payoffs = [intrinsic(s, a.K, kind) for s in stock]
    values = [np.zeros(i + 1) for i in range(a.n + 1)]
    continuation = [np.zeros(i + 1) for i in range(a.n + 1)]
    decisions = [np.full(i + 1, 'Continue', dtype=object) for i in range(a.n + 1)]
    values[-1] = payoffs[-1].copy()
    decisions[-1][:] = 'Expiry'
    if barrier is not None:
        knocked = stock[-1] >= barrier * (1 - 1e-12)
        values[-1][knocked] = 0
        decisions[-1][knocked] = 'Knocked out'
    for i in range(a.n - 1, -1, -1):
        continuation[i] = ((1 - p) * values[i + 1][:-1] + p * values[i + 1][1:]) / R
        values[i] = continuation[i].copy()
        if i in allowed:
            values[i] = np.maximum(values[i], payoffs[i])
            tol = 1e-10 * np.maximum(1, values[i])
            decisions[i][payoffs[i] > continuation[i] + tol] = 'Exercise'
            decisions[i][np.abs(payoffs[i] - continuation[i]) <= tol] = 'Indifferent'
        if barrier is not None:
            knocked = stock[i] >= barrier * (1 - 1e-12)
            values[i][knocked] = 0
            decisions[i][knocked] = 'Knocked out'
    return dict(stock=stock, values=values, payoff=payoffs, continuation=continuation,
                decisions=decisions, dt=dt, u=u, d=d, R=R, p=p, allowed=allowed,
                kind=kind, style=style, barrier=barrier)


def price(a, kind='Call', style='European', dates=()):
    """O(n) memory backward induction for convergence experiments."""
    dt, u, d, R, p = a.terms()
    allowed = exercise_steps(a.n, style, dates)
    j = np.arange(a.n + 1)
    v = intrinsic(a.S * u ** j * d ** (a.n - j), a.K, kind)
    for i in range(a.n - 1, -1, -1):
        v = ((1 - p) * v[:-1] + p * v[1:]) / R
        if i in allowed:
            j = np.arange(i + 1)
            v = np.maximum(v, intrinsic(a.S * u ** j * d ** (i - j), a.K, kind))
    return float(v[0])


def hedge(a, tree, i, j):
    if i >= a.n:
        raise ValueError('Replication applies before expiry.')
    if tree['decisions'][i][j] in ('Exercise', 'Knocked out'):
        raise ValueError('The contract settles at this node; a continuation hedge is not an active-option hedge.')
    S = float(tree['stock'][i][j])
    vd, vu = tree['values'][i + 1][j:j + 2]
    H = exp(-a.q * tree['dt']) * (vu - vd) / (S * (tree['u'] - tree['d']))
    B = H * S - tree['continuation'][i][j]
    return float(H), float(B)


def actual_probability(a, mu_price):
    dt, u, d, _, _ = a.terms()
    pa = (exp(mu_price * dt) - d) / (u - d)
    if not 0 <= pa <= 1:
        raise ValueError('This expected price growth is outside the tree: its actual up probability must lie between 0 and 1.')
    return pa


def distribution(n, p):
    if not 0 <= p <= 1:
        raise ValueError('Probability must be between zero and one.')
    return np.array([comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(n + 1)])


def node_return(a, tree, i, j, pa):
    if not 0 <= pa <= 1:
        raise ValueError('Probability must be between zero and one.')
    V = float(tree['values'][i][j])
    if V <= 1e-10:
        return None
    vd, vu = tree['values'][i + 1][j:j + 2]
    H, B = hedge(a, tree, i, j)
    S = float(tree['stock'][i][j])
    gross = float(((1 - pa) * vd + pa * vu) / V)
    stock_gross = exp(a.q * tree['dt']) * (pa * tree['u'] + (1 - pa) * tree['d'])
    return dict(gross=gross, simple=gross - 1, log_rate=log(gross) / tree['dt'] if gross > 0 else None,
                stock_gross=stock_gross, leverage=H * S / V,
                predicted=tree['R'] + H * S / V * (stock_gross - tree['R']), H=H, B=B)


def black_scholes(a, kind='Call'):
    N = lambda x: (1 + erf(x / sqrt(2))) / 2
    if a.K == 0:
        return a.S * exp(-a.q * a.T) if kind == 'Call' else 0.0
    if a.sigma <= 0:
        f = a.S * exp((a.r - a.q) * a.T)
        return float(intrinsic(f, a.K, kind)) * exp(-a.r * a.T)
    d1 = (log(a.S / a.K) + (a.r - a.q + a.sigma ** 2 / 2) * a.T) / (a.sigma * sqrt(a.T))
    d2 = d1 - a.sigma * sqrt(a.T)
    if kind == 'Call':
        return a.S * exp(-a.q * a.T) * N(d1) - a.K * exp(-a.r * a.T) * N(d2)
    return a.K * exp(-a.r * a.T) * N(-d2) - a.S * exp(-a.q * a.T) * N(-d1)
