"""
signals.py — EWMA expected returns, momentum signal, signal blending,
             and covariance estimation with Ledoit-Wolf shrinkage.

Signal stack (additive, blended into final mu):
  1. EWMA (lambda=0.94) — primary quantitative signal
  2. N-tick momentum — supplement
  3. Fundamental prior (FundamentalScorer) — quality/value anchor, strongest early
  4. 52-week mean-reversion (FundamentalScorer) — price-range mean reversion
  5. LLM overlay — blended when available

Covariance estimation hierarchy:
  1. Ledoit-Wolf shrinkage (sklearn) when >= LW_MIN_OBS observations
  2. Factor model (beta + sector) blended with shrinkage
  3. Diagonal fallback (per-ticker variance)
"""
import math
from collections import defaultdict
from typing import Optional

import numpy as np

from src.log import vprint

EWMA_LAMBDA   = 0.94    # RiskMetrics standard decay factor
MOMENTUM_N    = 10      # lookback window for momentum
LLM_ALPHA     = 0.30    # weight on LLM signal when available
MIN_HISTORY   = 5       # min ticks before we trust EWMA
LW_MIN_OBS    = 20      # min observations before using Ledoit-Wolf

# Fundamental prior signal scale: controls how strongly fundamentals
# influence expected returns. At 0.002 max, fundamentals contribute only
# a tiny ±0.2% tiebreaker — the fundamental prior is deliberately weak so
# it doesn't override EWMA momentum signals or CA event boosts. Its role is
# to provide a quality-informed initial portfolio construction when EWMA ≈ 0,
# without over-concentrating into quality stocks that may not outperform.
FUND_SCALE    = 0.002   # normalised fundamental prior max magnitude
RANGE_52W_SCALE = 0.002 # max contribution from 52-week price range signal


class FundamentalScorer:
    """
    Computes a static quality/value composite score per ticker from fundamentals.

    The composite acts as a *prior* on expected returns before we have price data.
    It biases initial portfolio construction toward quality stocks (high ROE, low P/E,
    positive EPS growth, BUY ratings, low ESG risk) and fades in relative importance
    as EWMA and momentum signals develop over the first 20+ ticks.

    Additionally provides a dynamic 52-week range signal that identifies mean-reversion
    opportunities (stocks near 52w lows) and overextended stocks (near 52w highs).

    Composite formula (all inputs z-scored cross-sectionally, then weighted):
        score = +0.25*(1/PE)    # earnings yield — lower P/E = more attractive
               +0.25*eps_growth # growth quality
               +0.20*roe        # capital efficiency
               +0.10*analyst    # consensus direction (BUY=+1, HOLD=0, SELL=-1)
               +0.10*div_yield  # income quality
               -0.05*debt_eq    # leverage penalty
               -0.05*esg_risk   # ESG risk penalty (higher numeric = higher risk)

    Score is scaled to [-FUND_SCALE, +FUND_SCALE] range for additive blending.
    """

    ANALYST_MAP = {"BUY": 1.0, "HOLD": 0.0, "SELL": -1.0, "STRONG BUY": 1.5,
                   "STRONG SELL": -1.5, "OUTPERFORM": 0.75, "UNDERPERFORM": -0.75}

    def __init__(self, fundamentals: dict, tickers: list[str]):
        self.tickers = tickers
        self._fundamentals = fundamentals

        # Extract raw features
        features: dict[str, dict] = {}
        for t in tickers:
            f = fundamentals.get(t, {})
            pe = float(f.get("pe_ratio", 20.0))
            eps_g = float(f.get("eps_growth_yoy", 0.0))
            roe = float(f.get("roe", 0.10))
            analyst_raw = f.get("analyst_rating", "HOLD")
            analyst = self.ANALYST_MAP.get(analyst_raw.upper(), 0.0)
            div_y = float(f.get("div_yield", 0.0))
            debt_eq = float(f.get("debt_to_equity", 0.5))
            esg = float(f.get("esg_score", 30.0))
            features[t] = {
                "earnings_yield": 1.0 / max(pe, 1.0),
                "eps_growth": eps_g,
                "roe": roe,
                "analyst": analyst,
                "div_yield": div_y,
                "debt_eq": debt_eq,
                "esg_risk": esg / 100.0,   # normalise ESG to [0,1]
            }

        # Z-score each feature cross-sectionally
        feature_keys = ["earnings_yield", "eps_growth", "roe", "analyst",
                        "div_yield", "debt_eq", "esg_risk"]
        weights = [0.25, 0.25, 0.20, 0.10, 0.10, -0.05, -0.05]

        z_scores: dict[str, dict[str, float]] = {t: {} for t in tickers}
        for key in feature_keys:
            vals = np.array([features[t][key] for t in tickers])
            mean_v = float(np.mean(vals))
            std_v  = float(np.std(vals))
            if std_v < 1e-10:
                std_v = 1.0
            for t in tickers:
                z_scores[t][key] = (features[t][key] - mean_v) / std_v

        # Weighted composite z-score
        raw_scores: dict[str, float] = {}
        for t in tickers:
            raw_scores[t] = sum(
                w * z_scores[t][k]
                for w, k in zip(weights, feature_keys)
            )

        # Scale to [-FUND_SCALE, +FUND_SCALE]
        max_abs = max(abs(v) for v in raw_scores.values()) or 1.0
        self._fund_mu: dict[str, float] = {
            t: float(raw_scores[t] / max_abs * FUND_SCALE)
            for t in tickers
        }

        # Pre-store 52w range data and beta
        self._high_52w: dict[str, float] = {}
        self._low_52w: dict[str, float] = {}
        self._beta: dict[str, float] = {}
        for t in tickers:
            f = fundamentals.get(t, {})
            self._high_52w[t] = float(f.get("high_52w", 0.0))
            self._low_52w[t]  = float(f.get("low_52w",  0.0))
            self._beta[t]     = float(f.get("beta",     1.0))

        # Print summary at init
        ranked = sorted(self._fund_mu.items(), key=lambda x: -x[1])
        top5 = [(t, f"{s:+.4f}") for t, s in ranked[:5]]
        bot5 = [(t, f"{s:+.4f}") for t, s in ranked[-5:]]
        vprint(f"[FundamentalScorer] Top 5 quality: {top5}")
        vprint(f"[FundamentalScorer] Bot 5 quality: {bot5}")

    def fundamental_mu(self) -> dict[str, float]:
        """Static quality prior on expected returns. Always call this."""
        return dict(self._fund_mu)

    def price_vs_52w(self, prices: dict[str, float]) -> dict[str, float]:
        """
        Mean-reversion signal based on where the current price sits in its
        52-week high-low range.

        range_pos = (price - low_52w) / (high_52w - low_52w)
          → 0 = at 52w low (buy signal: +RANGE_52W_SCALE)
          → 1 = at 52w high (sell signal: -RANGE_52W_SCALE)
          → 0.5 = midpoint (zero signal)

        Signal: RANGE_52W_SCALE * (0.5 - range_pos)
        """
        result: dict[str, float] = {}
        for t in self.tickers:
            hi = self._high_52w[t]
            lo = self._low_52w[t]
            price = prices.get(t, 0.0)
            if hi > lo and price > 0:
                pos = min(max((price - lo) / (hi - lo), 0.0), 1.0)
                result[t] = RANGE_52W_SCALE * (0.5 - pos)
            else:
                result[t] = 0.0
        return result

    def get_beta(self, ticker: str) -> float:
        return self._beta.get(ticker, 1.0)

    def beta_vector(self, tickers: list[str]) -> list[float]:
        return [self._beta.get(t, 1.0) for t in tickers]

    def sector_groups(self, tickers: list[str]) -> dict[str, list[int]]:
        """
        Returns {sector_name: [indices in tickers list]} for use in optimizer
        sector diversification constraints.
        """
        groups: dict[str, list[int]] = {}
        for i, t in enumerate(tickers):
            sector = self._fundamentals.get(t, {}).get("sector", "Unknown")
            groups.setdefault(sector, []).append(i)
        return groups

    def esg_weight_caps(self, cap_high: float = 0.08, cap_very_high: float = 0.05,
                        esg_high_threshold: int = 60,
                        esg_very_high_threshold: int = 75) -> dict[str, float]:
        """
        Returns per-ticker permanent weight caps based on ESG risk score.
        High-ESG-risk stocks are capped to limit regulatory fine exposure.
        """
        caps: dict[str, float] = {}
        for t in self.tickers:
            esg = int(self._fundamentals.get(t, {}).get("esg_score", 0))
            if esg >= esg_very_high_threshold:
                caps[t] = cap_very_high
            elif esg >= esg_high_threshold:
                caps[t] = cap_high
        return caps


class VolRegime:
    """
    Tracks rolling realised volatility and classifies current regime.

    regime: 'low' | 'normal' | 'high'

    Used by main.py to dynamically scale gamma and rebalance frequency.
    """

    # Thresholds relative to the long-run vol baseline
    HIGH_VOL_MULT = 1.8   # current vol > 1.8× baseline → HIGH
    LOW_VOL_MULT  = 0.6   # current vol < 0.6× baseline → LOW

    def __init__(self, short_window: int = 10, long_window: int = 40,
                 gamma_multipliers: tuple = (0.6, 1.0, 2.5),
                 rebal_multipliers: tuple = (0, 0, 5)):
        self._short = short_window
        self._long  = long_window
        self._gamma_mult  = {"low": gamma_multipliers[0], "normal": gamma_multipliers[1],
                              "high": gamma_multipliers[2]}
        self._rebal_mult  = {"low": rebal_multipliers[0], "normal": rebal_multipliers[1],
                              "high": rebal_multipliers[2]}
        self._portfolio_returns: list[float] = []

    def update(self, portfolio_value: float) -> None:
        """Call each tick with the current total portfolio value."""
        self._portfolio_returns.append(portfolio_value)

    @property
    def regime(self) -> str:
        """Return 'low', 'normal', or 'high'."""
        values = self._portfolio_returns
        if len(values) < self._long + 1:
            return "normal"

        log_rets = np.diff(np.log(np.array(values, dtype=float)))
        log_rets = log_rets[np.isfinite(log_rets)]

        if len(log_rets) < self._short:
            return "normal"

        short_vol = float(np.std(log_rets[-self._short:]))
        long_vol  = float(np.std(log_rets[-self._long:]))

        if long_vol < 1e-12:
            return "normal"

        ratio = short_vol / long_vol
        if ratio > self.HIGH_VOL_MULT:
            return "high"
        if ratio < self.LOW_VOL_MULT:
            return "low"
        return "normal"

    @property
    def gamma_multiplier(self) -> float:
        """Scale factor for MVO gamma based on volatility regime."""
        return self._gamma_mult[self.regime]

    @property
    def rebalance_multiplier(self) -> int:
        """How many extra ticks to skip between rebalances in high vol."""
        return self._rebal_mult[self.regime]


class SignalEngine:
    """
    Maintains rolling EWMA state and price history for all tickers.
    Call update(prices) every tick.
    """

    def __init__(self, tickers: list[str], fundamentals: Optional[dict] = None,
                 params: Optional[dict] = None):
        self.tickers = tickers
        self._fundamentals = fundamentals or {}
        # Accept params dict — extract signal-related knobs
        _p = params or {}
        self._ewma_lambda      = float(_p.get("EWMA_LAMBDA",       EWMA_LAMBDA))
        self._momentum_n       = int(  _p.get("MOMENTUM_N",        MOMENTUM_N))
        self._llm_alpha        = float(_p.get("LLM_ALPHA",         LLM_ALPHA))
        self._min_history      = int(  _p.get("MIN_HISTORY",       MIN_HISTORY))
        self._blend_cum_sharpe = float(_p.get("BLEND_CUM_SHARPE",  0.4))
        self._blend_ewma       = float(_p.get("BLEND_EWMA",        0.4))
        self._blend_momentum   = float(_p.get("BLEND_MOMENTUM",    0.2))
        self._cum_sharpe_scale = float(_p.get("CUM_SHARPE_SCALE",  0.003))

        # EWMA state: {ticker: float}
        self._ewma: dict[str, float] = {t: 0.0 for t in tickers}
        # Price history: {ticker: [price0, price1, ...]}
        self._price_history: dict[str, list[float]] = defaultdict(list)
        # Log-return history: {ticker: [r0, r1, ...]}
        self._return_history: dict[str, list[float]] = defaultdict(list)
        self._tick_count = 0

        # Pre-compute factor structures from fundamentals
        self._beta = self._extract_beta()
        self._sector_dummies = self._build_sector_dummies()

    # ------------------------------------------------------------------
    # Fundamentals helpers
    # ------------------------------------------------------------------
    def _extract_beta(self) -> np.ndarray:
        """Beta vector for all tickers (market factor loading)."""
        betas = []
        for t in self.tickers:
            fund = self._fundamentals.get(t, {})
            betas.append(float(fund.get("beta", 1.0)))
        return np.array(betas, dtype=float)

    def _build_sector_dummies(self) -> np.ndarray:
        """
        Sector dummy matrix: shape (n_tickers, n_sectors).
        One-hot encoding of sector membership.
        """
        sectors = sorted(set(
            self._fundamentals.get(t, {}).get("sector", "Unknown")
            for t in self.tickers
        ))
        sector_idx = {s: i for i, s in enumerate(sectors)}
        n = len(self.tickers)
        k = len(sectors)
        D = np.zeros((n, k), dtype=float)
        for i, t in enumerate(self.tickers):
            s = self._fundamentals.get(t, {}).get("sector", "Unknown")
            D[i, sector_idx[s]] = 1.0
        return D  # (n, k)

    # ------------------------------------------------------------------
    # Update state each tick
    # ------------------------------------------------------------------
    def update(self, prices: dict[str, float]) -> None:
        """Ingest new prices and update EWMA + history."""
        for ticker in self.tickers:
            p = prices.get(ticker)
            if p is None or p <= 0:
                continue

            history = self._price_history[ticker]

            if history:
                    prev_p = history[-1]
                    if prev_p > 0:
                        log_r = math.log(p / prev_p)
                        self._return_history[ticker].append(log_r)
                        # Update EWMA
                        self._ewma[ticker] = (
                            self._ewma_lambda * log_r
                            + (1 - self._ewma_lambda) * self._ewma[ticker]
                        )

            history.append(p)

        self._tick_count += 1

    # ------------------------------------------------------------------
    # Signal computation
    # ------------------------------------------------------------------
    def ewma_returns(self) -> dict[str, float]:
        """Return current EWMA expected return estimate per ticker."""
        return dict(self._ewma)

    def momentum(self, n: int | None = None) -> dict[str, float]:
        """
        N-tick momentum: (P_t - P_{t-N}) / P_{t-N}
        Returns 0 if insufficient history.
        """
        if n is None:
            n = self._momentum_n
        result = {}
        for ticker in self.tickers:
            hist = self._price_history[ticker]
            if len(hist) >= n + 1:
                p_now  = hist[-1]
                p_prev = hist[-(n + 1)]
                if p_prev > 0:
                    result[ticker] = (p_now - p_prev) / p_prev
                else:
                    result[ticker] = 0.0
            else:
                result[ticker] = 0.0
        return result

    def cumulative_sharpe(self, scale: float | None = None) -> dict[str, float]:
        """
        Compute cumulative realized per-tick Sharpe ratio since start for each ticker,
        scaled to the same magnitude as EWMA signals.

        This signal identifies stocks with consistently positive risk-adjusted returns
        across the entire session so far — it is robust to CA-event price spikes
        (a spike followed by flat returns degrades the cumulative Sharpe over time,
        unlike EWMA which stays elevated for many ticks after the spike).

        Returns: {ticker: scale * sharpe} — values typically in [-0.01, +0.01].
        """
        if scale is None:
            scale = self._cum_sharpe_scale
        result = {}
        for ticker in self.tickers:
            rets = self._return_history.get(ticker, [])
            if len(rets) < 5:
                result[ticker] = 0.0
                continue
            arr = np.array(rets, dtype=float)
            mu = float(np.mean(arr))
            sigma = float(np.std(arr))
            sh = mu / sigma if sigma > 1e-10 else 0.0
            result[ticker] = scale * sh
        return result

    def blended_returns(
        self,
        llm_mu: Optional[dict[str, float]] = None,
        ca_boosts: Optional[dict[str, float]] = None,
        fundamental_prior: Optional[dict[str, float]] = None,
        price_52w_signal: Optional[dict[str, float]] = None,
    ) -> dict[str, float]:
        """
        Blend all signals into a final expected return per ticker.

        Primary signal stack:
          quant_mu = 0.4 * cumulative_sharpe + 0.4 * ewma + 0.2 * momentum

        The cumulative Sharpe (realized risk-adjusted return since session start)
        is robust to short-term CA event spikes: a spike followed by mediocre
        returns degrades the Sharpe over time, naturally de-emphasising the
        post-spike ticker. EWMA and momentum complement with recency.

        LLM overlay (when available):
          mu = alpha*llm_mu + (1-alpha)*quant_mu

        CA boosts applied additively on top (calibrated to same magnitude).
        """
        ewma   = self.ewma_returns()
        mom    = self.momentum()
        cum_sh = self.cumulative_sharpe()

        result = {}
        for ticker in self.tickers:
            # Blend: cumulative Sharpe + EWMA + momentum
            quant_mu = (
                self._blend_cum_sharpe * cum_sh.get(ticker, 0.0)
                + self._blend_ewma     * ewma.get(ticker, 0.0)
                + self._blend_momentum * mom.get(ticker, 0.0)
            )

            # Fundamental quality prior (disabled by default — pass None to skip)
            if fundamental_prior:
                quant_mu += fundamental_prior.get(ticker, 0.0)

            # 52-week range mean-reversion (disabled by default — pass None to skip)
            if price_52w_signal:
                quant_mu += price_52w_signal.get(ticker, 0.0)

            # LLM overlay blend
            if llm_mu and ticker in llm_mu:
                mu = self._llm_alpha * llm_mu[ticker] + (1 - self._llm_alpha) * quant_mu
            else:
                mu = quant_mu

            # Corporate action signal boosts (already decayed by caller)
            if ca_boosts and ticker in ca_boosts:
                mu += ca_boosts[ticker]

            result[ticker] = mu

        return result

    def covariance_matrix(self, tickers: list[str]) -> list[list[float]]:
        """
        Estimate covariance matrix using a hierarchy:

        1. Ledoit-Wolf shrinkage (sklearn) when >= LW_MIN_OBS observations
           — analytically optimal shrinkage intensity, no free parameters
        2. Factor covariance (beta + sector) blended with shrinkage when
           fundamentals are available
        3. Diagonal fallback (per-ticker variance or default)

        Returns n×n matrix (list of lists) for the given ticker order.
        """
        n = len(tickers)

        # Collect aligned return vectors
        lengths = [len(self._return_history[t]) for t in tickers]
        min_len = min(lengths) if lengths else 0

        if min_len >= LW_MIN_OBS:
            cov = self._ledoit_wolf_cov(tickers, min_len)
        elif min_len >= max(MIN_HISTORY, n + 2):
            # Enough for sample cov but not LW — use factor model blend
            cov = self._factor_cov(tickers, min_len)
        else:
            cov = self._diagonal_fallback(tickers)

        return cov.tolist()

    def _ledoit_wolf_cov(self, tickers: list[str], min_len: int) -> np.ndarray:
        """
        Ledoit-Wolf analytical shrinkage toward a structured target.
        Uses sklearn's LedoitWolf estimator.
        Blends with factor covariance for additional stability.
        """
        try:
            from sklearn.covariance import LedoitWolf

            mat = np.array([
                self._return_history[t][-min_len:]
                for t in tickers
            ])  # shape (n, T)

            lw = LedoitWolf(assume_centered=False)
            lw.fit(mat.T)           # sklearn expects (T, n)
            cov_lw = lw.covariance_  # (n, n)

            # Blend with factor covariance: 70% LW + 30% factor
            if self._fundamentals:
                cov_factor = self._factor_cov(tickers, min_len)
                cov = 0.7 * cov_lw + 0.3 * cov_factor
            else:
                cov = cov_lw

            # Small ridge for numerical stability
            cov += np.eye(len(tickers)) * 1e-8
            return cov

        except ImportError:
            # sklearn not available — fall back to sample cov
            return self._sample_cov(tickers, min_len)

    def _sample_cov(self, tickers: list[str], min_len: int) -> np.ndarray:
        """Plain sample covariance with ridge."""
        mat = np.array([
            self._return_history[t][-min_len:]
            for t in tickers
        ])
        cov = np.cov(mat)
        cov += np.eye(len(tickers)) * 1e-8
        return cov

    def _factor_cov(self, tickers: list[str], min_len: int) -> np.ndarray:
        """
        Factor covariance: Σ = B × F_mkt × B' + D × F_sec × D' + Ψ

        where:
          B       = beta vector (market factor loading, n×1)
          D       = sector dummies (n×k)
          F_mkt   = market factor variance (scalar)
          F_sec   = sector factor covariance (k×k diagonal)
          Ψ       = diagonal idiosyncratic variance

        Uses beta and sector from fundamentals.
        Falls back to diagonal if fundamentals are missing.
        """
        n = len(tickers)

        # Get ticker indices in the full tickers list
        ticker_idx = {t: i for i, t in enumerate(self.tickers)}
        idx = [ticker_idx[t] for t in tickers]

        beta_sub = self._beta[idx]                    # (n,)
        sector_sub = self._sector_dummies[idx, :]     # (n, k)

        # Estimate market factor variance from return histories
        # Use average of all ticker returns as a proxy for market
        all_rets = np.array([
            self._return_history[t][-min_len:]
            for t in tickers
        ])  # (n, T)

        market_proxy = np.mean(all_rets, axis=0)   # (T,)
        sigma_mkt_sq = float(np.var(market_proxy)) + 1e-8

        # Market factor contribution: beta * beta^T * sigma_mkt^2
        cov_mkt = np.outer(beta_sub, beta_sub) * sigma_mkt_sq  # (n, n)

        # Sector factor contribution: D * F_sec * D^T
        # Estimate per-sector variance from sector-average returns
        n_sectors = sector_sub.shape[1]
        F_sec = np.zeros((n_sectors, n_sectors))
        for j in range(n_sectors):
            members = np.where(sector_sub[:, j] > 0)[0]
            if len(members) > 0:
                sec_rets = np.mean(all_rets[members, :], axis=0)
                F_sec[j, j] = float(np.var(sec_rets)) + 1e-8
        cov_sec = sector_sub @ F_sec @ sector_sub.T  # (n, n)

        # Idiosyncratic variance: residuals after removing factor contributions
        Psi = np.zeros(n)
        for i, t in enumerate(tickers):
            total_var = float(np.var(all_rets[i])) + 1e-8
            factor_var = beta_sub[i]**2 * sigma_mkt_sq
            Psi[i] = max(total_var - factor_var, 1e-8)

        cov_factor = cov_mkt + cov_sec + np.diag(Psi)
        return cov_factor

    def _diagonal_fallback(self, tickers: list[str]) -> np.ndarray:
        """Diagonal matrix based on per-ticker variance; used when history is short."""
        n = len(tickers)
        cov = np.zeros((n, n))
        for i, ticker in enumerate(tickers):
            rets = self._return_history[ticker]
            if len(rets) >= 2:
                cov[i, i] = float(np.var(rets)) + 1e-8
            else:
                cov[i, i] = (0.002) ** 2  # default 0.2% per-tick vol
        return cov

    # ------------------------------------------------------------------
    # Price history accessor (used by corporate_actions for split adjust)
    # ------------------------------------------------------------------
    @property
    def price_histories(self) -> dict[str, list[float]]:
        return self._price_history

    def has_sufficient_history(self) -> bool:
        return self._tick_count >= self._min_history
