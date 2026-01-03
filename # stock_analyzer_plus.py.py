# stock_analyzer_plus.py
# 7대 지표 기반 주식 종합 분석기 + ADX/ATR/VWAP/CMF + 레짐 가중치 + 간단 백테스트
# Libraries: yfinance, pandas_ta, matplotlib (+ numpy, pandas)

from __future__ import annotations

import os
import sys
import argparse
from dataclasses import dataclass
import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import matplotlib.pyplot as plt

# "안티그래비티" 요청 반영(표준 라이브러리) - 실행에 영향 없음
try:
    import antigravity  # noqa: F401
except Exception:
    pass


# -----------------------------
# Config / Data structures
# -----------------------------
@dataclass
class ScoreConfig:
    # 레짐 판별
    adx_threshold: float = 25.0  # ADX >= 25 -> TREND, else RANGE

    # 레짐별 가중치 조정 (기본 1.0)
    trend_boost: float = 1.15     # 추세장: 추세계열 가중치 * 1.15
    meanrev_cut: float = 0.85     # 추세장: 되돌림계열 가중치 * 0.85
    trend_cut: float = 0.85       # 횡보장: 추세계열 가중치 * 0.85
    meanrev_boost: float = 1.15   # 횡보장: 되돌림계열 가중치 * 1.15

    # ATR 리스크 패널티 기준(ATR/Close)
    atr_pct_threshold: float = 0.05  # 5% 초과면 변동성 리스크로 감점

    # 반등 탐지 lookback
    rebound_lookback: int = 10

    # 볼린저 하단 근처 기준 (BBP 있으면 이 값 이하)
    bbp_lower_threshold: float = 0.20

    # 백테스트 설정
    score_threshold: float = 70.0
    horizon_days: int = 5


@dataclass
class ScoreResult:
    raw_points: float
    score_100: float
    grade: str
    regime: str
    details: dict


# -----------------------------
# Utilities
# -----------------------------
def download_ohlcv(ticker: str, interval: str, period: str) -> pd.DataFrame:
    df = yf.download(
        ticker,
        interval=interval,
        period=period,
        auto_adjust=False,
        progress=False,
    )
    if df is None or df.empty:
        raise RuntimeError(f"데이터를 받지 못했습니다. 티커({ticker})/네트워크를 확인하세요.")

    # yfinance가 MultiIndex로 내려오는 경우 대비
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    required = {"Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(set(df.columns)):
        raise RuntimeError(f"필수 컬럼(OHLCV)이 부족합니다. 현재 컬럼: {list(df.columns)}")

    df = df.dropna(subset=["High", "Low", "Close", "Volume"]).copy()
    return df


def _pick_col(df: pd.DataFrame, prefix: str) -> str:
    cols = [c for c in df.columns if str(c).startswith(prefix)]
    if not cols:
        raise RuntimeError(f"컬럼({prefix}*)을 찾지 못했습니다. pandas_ta 버전/계산 여부를 확인하세요.")
    return cols[0]


def rebound_from_30_to_40(series: pd.Series, lookback: int = 10) -> pd.Series:
    """
    '최근 lookback 구간 중 최저 <= 30' AND '현재 >= 40' -> True
    (벡터화된 시리즈 반환)
    """
    s = series.astype(float)
    roll_min = s.rolling(lookback, min_periods=lookback).min()
    return (roll_min <= 30.0) & (s >= 40.0)


def near_lower_bollinger(
    close: pd.Series,
    bbl: pd.Series,
    bbu: pd.Series,
    bbp: pd.Series | None,
    bbp_lower_threshold: float = 0.20,
) -> pd.Series:
    """
    '볼린저 하단 부근' 판정(벡터):
    - BB% (BBP)가 있으면 bbp <= threshold
    - 없으면 밴드 폭 하단 10% 이내
    """
    valid = bbu > bbl
    if bbp is not None:
        return valid & bbp.notna() & (bbp <= bbp_lower_threshold)

    # 대체 기준: 하단 + 10% 폭
    threshold = bbl + 0.10 * (bbu - bbl)
    return valid & close.notna() & threshold.notna() & (close <= threshold)


# -----------------------------
# Indicators
# -----------------------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    기존 지표:
    - MA(20,60,120), Ichimoku, MACD
    - Bollinger(20,2), OBV
    - RSI(14), MFI(14)

    추가:
    - ADX(14), ATR(14), VWAP, CMF(20)
    """
    out = df.copy()

    # 이동평균
    out["MA20"] = ta.sma(out["Close"], length=20)
    out["MA60"] = ta.sma(out["Close"], length=60)
    out["MA120"] = ta.sma(out["Close"], length=120)

    # Bollinger Bands
    bb = ta.bbands(out["Close"], length=20, std=2)
    out = out.join(bb)

    # MACD
    macd = ta.macd(out["Close"], fast=12, slow=26, signal=9)
    out = out.join(macd)

    # RSI / MFI
    out["RSI14"] = ta.rsi(out["Close"], length=14)
    out["MFI14"] = ta.mfi(out["High"], out["Low"], out["Close"], out["Volume"], length=14)

    # OBV + MA20
    out["OBV"] = ta.obv(out["Close"], out["Volume"])
    out["OBV_MA20"] = out["OBV"].rolling(20).mean()

    # Ichimoku (tuple 반환이 일반적)
    ichi = ta.ichimoku(out["High"], out["Low"], out["Close"])
    if isinstance(ichi, tuple) and len(ichi) >= 2:
        ichi_line_df, ichi_span_df = ichi[0], ichi[1]
        out = out.join(ichi_line_df)
        out = out.join(ichi_span_df.reindex(out.index))
    else:
        out = out.join(ichi)

    # ADX(14)
    adx_df = ta.adx(out["High"], out["Low"], out["Close"], length=14)
    out = out.join(adx_df)

    # ATR(14) - 컬럼명은 환경에 따라 달라질 수 있어 통일 저장
    atr = ta.atr(out["High"], out["Low"], out["Close"], length=14)
    out["ATR14"] = atr.astype(float)

    # VWAP (일봉에서도 누적 VWAP 형태로 계산됨)
    vwap = ta.vwap(out["High"], out["Low"], out["Close"], out["Volume"])
    # 이름 통일
    out["VWAP"] = vwap.astype(float)

    # CMF(20)
    cmf = ta.cmf(out["High"], out["Low"], out["Close"], out["Volume"], length=20)
    out["CMF20"] = cmf.astype(float)

    return out


# -----------------------------
# Scoring (Regime-aware)
# -----------------------------
def compute_score_series(df: pd.DataFrame, cfg: ScoreConfig) -> pd.DataFrame:
    """
    레짐(ADX) 기반 가중치 자동 변경 + 점수 시계열 생성 (백테스트/차트용)

    기존 룰(요청한 점수판) 유지 + 확장:
    - Ichimoku cloud 위 +20, 아래 -20
    - MACD Hist > 0 +15
    - RSI or MFI (30이하->40이상 반등) +20
    - 볼린저 하단 부근 +15
    - OBV > OBV_MA20 +15

    추가(확장 점수):
    - Close > VWAP +7
    - CMF20 > 0 +8
    - ATR% (ATR14/Close) > threshold -> -5 (변동성 리스크)
    """
    # 컬럼 탐색
    span_a_col = _pick_col(df, "ISA_")
    span_b_col = _pick_col(df, "ISB_")
    macdh_col = _pick_col(df, "MACDh_")
    bbl_col = _pick_col(df, "BBL_")
    bbu_col = _pick_col(df, "BBU_")
    bbp_col = None
    for c in df.columns:
        if str(c).startswith("BBP_"):
            bbp_col = c
            break
    adx_col = _pick_col(df, "ADX_")

    needed = [
        "Close", "High", "Low", "Volume",
        span_a_col, span_b_col, macdh_col,
        "RSI14", "MFI14",
        bbl_col, bbu_col,
        "OBV", "OBV_MA20",
        "ATR14", "VWAP", "CMF20",
        adx_col,
    ]
    tmp = df.copy()
    # BBP는 선택
    if bbp_col is not None:
        needed.append(bbp_col)

    valid = tmp.dropna(subset=[c for c in needed if c in tmp.columns]).copy()
    if valid.empty:
        raise RuntimeError("유효한 지표 계산 구간이 없습니다. 데이터 기간을 늘려보세요.")

    close = valid["Close"].astype(float)

    # Ichimoku cloud top/bottom
    span_a = valid[span_a_col].astype(float)
    span_b = valid[span_b_col].astype(float)
    cloud_top = pd.concat([span_a, span_b], axis=1).max(axis=1)
    cloud_bot = pd.concat([span_a, span_b], axis=1).min(axis=1)

    cond_cloud_above = close > cloud_top
    cond_cloud_below = close < cloud_bot

    # MACD hist
    macdh = valid[macdh_col].astype(float)
    cond_macd_pos = macdh > 0

    # Rebound (RSI or MFI)
    rsi = valid["RSI14"].astype(float)
    mfi = valid["MFI14"].astype(float)
    cond_rsi_reb = rebound_from_30_to_40(rsi, lookback=cfg.rebound_lookback)
    cond_mfi_reb = rebound_from_30_to_40(mfi, lookback=cfg.rebound_lookback)
    cond_rebound = cond_rsi_reb | cond_mfi_reb

    # Bollinger lower proximity
    bbl = valid[bbl_col].astype(float)
    bbu = valid[bbu_col].astype(float)
    bbp = valid[bbp_col].astype(float) if bbp_col is not None else None
    cond_bb_lower = near_lower_bollinger(close, bbl, bbu, bbp, cfg.bbp_lower_threshold)

    # OBV
    obv = valid["OBV"].astype(float)
    obv_ma20 = valid["OBV_MA20"].astype(float)
    cond_obv = obv > obv_ma20

    # VWAP / CMF
    vwap = valid["VWAP"].astype(float)
    cmf20 = valid["CMF20"].astype(float)
    cond_vwap = close > vwap
    cond_cmf = cmf20 > 0

    # ATR risk
    atr14 = valid["ATR14"].astype(float)
    atr_pct = (atr14 / close).replace([np.inf, -np.inf], np.nan)
    cond_atr_risk = atr_pct > cfg.atr_pct_threshold

    # Regime by ADX
    adx = valid[adx_col].astype(float)
    is_trend = adx >= cfg.adx_threshold
    regime = np.where(is_trend, "TREND", "RANGE")

    # Base weights (요청 점수판 + 확장)
    W = {
        "ichimoku_above": 20.0,
        "ichimoku_below": -20.0,
        "macd_pos": 15.0,
        "rebound": 20.0,
        "bb_lower": 15.0,
        "obv": 15.0,
        "vwap": 7.0,
        "cmf": 8.0,
        "atr_risk": -5.0,
    }
    max_raw = W["ichimoku_above"] + W["macd_pos"] + W["rebound"] + W["bb_lower"] + W["obv"] + W["vwap"] + W["cmf"]  # 100
    min_raw = W["ichimoku_below"] + W["atr_risk"]  # -25

    # Regime factors
    # 추세 계열: ichimoku/macd/vwap/obv/cmf
    # 되돌림 계열: rebound/bb_lower
    trend_factor = np.where(is_trend, cfg.trend_boost, cfg.trend_cut)
    meanrev_factor = np.where(is_trend, cfg.meanrev_cut, cfg.meanrev_boost)

    # 점수 합산
    raw = np.zeros(len(valid), dtype=float)

    # Ichimoku above/below (penalty/bonus는 레짐 영향 X로 둠)
    raw += np.where(cond_cloud_above.values, W["ichimoku_above"], 0.0)
    raw += np.where(cond_cloud_below.values, W["ichimoku_below"], 0.0)

    # Trend group (레짐 반영)
    raw += np.where(cond_macd_pos.values, W["macd_pos"] * trend_factor, 0.0)
    raw += np.where(cond_obv.values, W["obv"] * trend_factor, 0.0)
    raw += np.where(cond_vwap.values, W["vwap"] * trend_factor, 0.0)
    raw += np.where(cond_cmf.values, W["cmf"] * trend_factor, 0.0)

    # Mean-reversion group (레짐 반영)
    raw += np.where(cond_rebound.values, W["rebound"] * meanrev_factor, 0.0)
    raw += np.where(cond_bb_lower.values, W["bb_lower"] * meanrev_factor, 0.0)

    # ATR risk penalty (레짐 영향 X)
    raw += np.where(cond_atr_risk.values, W["atr_risk"], 0.0)

    # Clip raw to expected bounds (레짐 가중치로 100 초과 가능 -> 상한 클리핑)
    raw = np.clip(raw, min_raw, max_raw)

    score_100 = (raw - min_raw) / (max_raw - min_raw) * 100.0
    score_100 = np.clip(score_100, 0.0, 100.0)

    def grade_from_score(s: float) -> str:
        if s >= 80:
            return "강력 매수"
        if s >= 65:
            return "매수"
        if s >= 45:
            return "관망"
        return "매도"

    grades = [grade_from_score(float(s)) for s in score_100]

    out = valid.copy()
    out["REGIME"] = regime
    out["ADX14"] = adx
    out["ATR_PCT"] = atr_pct
    out["RAW_POINTS"] = raw
    out["SCORE_100"] = score_100
    out["GRADE"] = grades

    # 조건 컬럼(디버깅/리포트/차트용)
    out["CLOUD_ABOVE"] = cond_cloud_above
    out["CLOUD_BELOW"] = cond_cloud_below
    out["MACD_POS"] = cond_macd_pos
    out["REB_REBOUND"] = cond_rebound
    out["BB_LOWER"] = cond_bb_lower
    out["OBV_OK"] = cond_obv
    out["VWAP_OK"] = cond_vwap
    out["CMF_OK"] = cond_cmf
    out["ATR_RISK"] = cond_atr_risk

    return out


def latest_score_detail(scored_df: pd.DataFrame, cfg: ScoreConfig) -> ScoreResult:
    last = scored_df.iloc[-1]
    score_100 = float(last["SCORE_100"])
    raw = float(last["RAW_POINTS"])
    regime = str(last["REGIME"])
    grade = str(last["GRADE"])

    # 간단 상세
    close = float(last["Close"])
    adx = float(last["ADX14"])
    atr_pct = float(last["ATR_PCT"]) if pd.notna(last["ATR_PCT"]) else np.nan
    vwap = float(last["VWAP"])
    cmf = float(last["CMF20"])

    detail = {
        "regime": f"{regime} (ADX14={adx:.1f}, threshold={cfg.adx_threshold})",
        "close": f"{close:,.0f}",
        "vwap": f"{vwap:,.0f} ({'상단' if close > vwap else '하단/근처'})",
        "cmf20": f"{cmf:+.3f} ({'매집 우위' if cmf > 0 else '분산/중립'})",
        "atr_pct": f"{(atr_pct * 100):.2f}% ({'리스크↑ 감점' if pd.notna(atr_pct) and atr_pct > cfg.atr_pct_threshold else '정상'})",
        "signals": {
            "ichimoku": ("구름대 위" if bool(last["CLOUD_ABOVE"]) else ("구름대 아래" if bool(last["CLOUD_BELOW"]) else "구름대 내부")),
            "macd_hist": ("양수" if bool(last["MACD_POS"]) else "음수/0"),
            "rebound": ("발생" if bool(last["REB_REBOUND"]) else "없음"),
            "boll_lower": ("하단부근" if bool(last["BB_LOWER"]) else "중립/상단"),
            "obv": ("우호" if bool(last["OBV_OK"]) else "약화"),
            "vwap": ("상단" if bool(last["VWAP_OK"]) else "하단/근처"),
            "cmf": ("양수" if bool(last["CMF_OK"]) else "음수/0"),
            "atr": ("리스크" if bool(last["ATR_RISK"]) else "정상"),
        },
    }

    return ScoreResult(raw_points=raw, score_100=score_100, grade=grade, regime=regime, details=detail)


# -----------------------------
# Summaries
# -----------------------------
def summarize_indicators(scored_df: pd.DataFrame) -> list[str]:
    """
    너무 길지 않게 핵심 상태 텍스트 요약
    """
    last = scored_df.iloc[-1]

    close = float(last["Close"])
    ma20 = float(last["MA20"]) if pd.notna(last.get("MA20", np.nan)) else np.nan
    ma60 = float(last["MA60"]) if pd.notna(last.get("MA60", np.nan)) else np.nan
    ma120 = float(last["MA120"]) if pd.notna(last.get("MA120", np.nan)) else np.nan

    rsi = float(last["RSI14"]) if pd.notna(last.get("RSI14", np.nan)) else np.nan
    mfi = float(last["MFI14"]) if pd.notna(last.get("MFI14", np.nan)) else np.nan
    adx = float(last["ADX14"]) if pd.notna(last.get("ADX14", np.nan)) else np.nan
    cmf = float(last["CMF20"]) if pd.notna(last.get("CMF20", np.nan)) else np.nan
    atr_pct = float(last["ATR_PCT"]) if pd.notna(last.get("ATR_PCT", np.nan)) else np.nan

    lines: list[str] = []

    # MA 정렬
    if all(pd.notna([ma20, ma60, ma120])):
        if close > ma20 > ma60 > ma120:
            lines.append("이동평균: 상승 정렬(20>60>120) & 종가 상단")
        elif close < ma20 < ma60 < ma120:
            lines.append("이동평균: 하락 정렬(20<60<120) & 종가 하단")
        else:
            lines.append("이동평균: 혼조/횡보(정렬 불명확)")

    # Ichimoku / MACD / Bollinger / OBV 등은 scored_df에 이미 신호가 있음
    ich = "구름대 위" if bool(last["CLOUD_ABOVE"]) else ("구름대 아래" if bool(last["CLOUD_BELOW"]) else "구름대 내부")
    lines.append(f"일목균형표: {ich}")

    lines.append(f"MACD: {'상승 모멘텀(히스토그램 양수)' if bool(last['MACD_POS']) else '약세 모멘텀(히스토그램 음수/0)'}")
    lines.append(f"볼린저: {'하단 부근(되돌림 가능)' if bool(last['BB_LOWER']) else '중립/상단'}")
    lines.append(f"OBV: {'거래량 흐름 우호' if bool(last['OBV_OK']) else '거래량 흐름 약화'}")
    lines.append(f"VWAP: {'상단(강세)' if bool(last['VWAP_OK']) else '하단/근처'}")
    lines.append(f"CMF(20): {cmf:+.3f} ({'매집 우위' if cmf > 0 else '분산/중립'})")

    # RSI / MFI
    if pd.notna(rsi):
        if rsi < 30:
            lines.append(f"RSI(14): 과매도({rsi:.1f})")
        elif rsi > 70:
            lines.append(f"RSI(14): 과매수({rsi:.1f})")
        else:
            lines.append(f"RSI(14): 중립({rsi:.1f})")
    if pd.notna(mfi):
        if mfi < 20:
            lines.append(f"MFI(14): 과매도({mfi:.1f})")
        elif mfi > 80:
            lines.append(f"MFI(14): 과매수({mfi:.1f})")
        else:
            lines.append(f"MFI(14): 중립({mfi:.1f})")

    # ADX / ATR
    if pd.notna(adx):
        lines.append(f"ADX(14): {adx:.1f} ({'추세장 가능' if adx >= 25 else '횡보 가능'})")
    if pd.notna(atr_pct):
        lines.append(f"ATR%: {(atr_pct*100):.2f}% ({'변동성 큼' if atr_pct > 0.05 else '보통'})")

    # 반등 신호
    lines.append(f"반등 신호(RSI/MFI): {'발생' if bool(last['REB_REBOUND']) else '없음'}")

    return lines


# -----------------------------
# Backtest
# -----------------------------
def simple_backtest(scored_df: pd.DataFrame, cfg: ScoreConfig) -> pd.DataFrame:
    """
    간단 백테스트:
    - signal: SCORE_100 >= X (cfg.score_threshold)
    - 다음날~N일(h=1..N) 수익률: Close[t+h] / Close[t] - 1
    - 통계: 표본수, 평균/중앙, 승률
    """
    close = scored_df["Close"].astype(float)
    score = scored_df["SCORE_100"].astype(float)
    signal = score >= cfg.score_threshold

    results = []
    for h in range(1, max(1, int(cfg.horizon_days)) + 1):
        fwd_ret = close.shift(-h) / close - 1.0
        sample = fwd_ret[signal & fwd_ret.notna()]
        if len(sample) == 0:
            results.append({
                "horizon_days": h,
                "signals": int(signal.sum()),
                "samples": 0,
                "avg_return_%": np.nan,
                "median_return_%": np.nan,
                "win_rate_%": np.nan,
            })
            continue

        win_rate = float((sample > 0).mean() * 100.0)
        results.append({
            "horizon_days": h,
            "signals": int(signal.sum()),
            "samples": int(sample.shape[0]),
            "avg_return_%": float(sample.mean() * 100.0),
            "median_return_%": float(sample.median() * 100.0),
            "win_rate_%": win_rate,
        })

    return pd.DataFrame(results)


# -----------------------------
# Plotting
# -----------------------------
def plot_indicators_and_score(scored_df: pd.DataFrame, ticker: str, cfg: ScoreConfig, bars: int = 250) -> None:
    """
    너무 복잡하지 않게 서브플롯 구성(6개):
    1) Price + MA + Bollinger + VWAP
    2) Ichimoku cloud + price
    3) MACD
    4) RSI + MFI
    5) OBV
    6) SCORE + ADX (둘 다 0~100 스케일) + 임계선 + 레짐 음영
    """
    span_a_col = _pick_col(scored_df, "ISA_")
    span_b_col = _pick_col(scored_df, "ISB_")
    macd_col = next((c for c in scored_df.columns if str(c).startswith("MACD_") and not str(c).startswith("MACDh_") and not str(c).startswith("MACDs_")), None)
    macds_col = _pick_col(scored_df, "MACDs_")
    macdh_col = _pick_col(scored_df, "MACDh_")
    bbl_col = _pick_col(scored_df, "BBL_")
    bbm_col = next((c for c in scored_df.columns if str(c).startswith("BBM_")), None)
    bbu_col = _pick_col(scored_df, "BBU_")

    plot_df = scored_df.copy().iloc[-bars:]

    fig, axes = plt.subplots(
        6, 1, figsize=(14, 16), sharex=True,
        gridspec_kw={"height_ratios": [3, 2, 2, 1.6, 1.6, 1.8]}
    )
    ax1, ax2, ax3, ax4, ax5, ax6 = axes

    # 1) Price + MA + BB + VWAP
    ax1.plot(plot_df.index, plot_df["Close"], label="Close", linewidth=1.2)
    for name in ["MA20", "MA60", "MA120"]:
        if name in plot_df.columns:
            ax1.plot(plot_df.index, plot_df[name], label=name, linewidth=1.0)

    ax1.plot(plot_df.index, plot_df["VWAP"], label="VWAP", linewidth=1.0, alpha=0.9)

    ax1.plot(plot_df.index, plot_df[bbu_col], label="BB Upper", linewidth=0.9, alpha=0.9)
    ax1.plot(plot_df.index, plot_df[bbl_col], label="BB Lower", linewidth=0.9, alpha=0.9)
    if bbm_col and bbm_col in plot_df.columns:
        ax1.plot(plot_df.index, plot_df[bbm_col], label="BB Mid", linewidth=0.9, alpha=0.8)
    ax1.fill_between(plot_df.index, plot_df[bbl_col], plot_df[bbu_col], alpha=0.08)

    ax1.set_title(f"{ticker} - Price / MA / Bollinger / VWAP (최근 {len(plot_df)}봉)")
    ax1.legend(loc="upper left", ncol=5, fontsize=9)
    ax1.grid(True, alpha=0.2)

    # 2) Ichimoku cloud + price
    ax2.plot(plot_df.index, plot_df["Close"], label="Close", linewidth=1.0)
    ax2.plot(plot_df.index, plot_df[span_a_col], label="Span A", linewidth=0.9, alpha=0.9)
    ax2.plot(plot_df.index, plot_df[span_b_col], label="Span B", linewidth=0.9, alpha=0.9)
    sa = plot_df[span_a_col].values
    sb = plot_df[span_b_col].values
    mask = ~np.isnan(sa) & ~np.isnan(sb)
    ax2.fill_between(plot_df.index, plot_df[span_a_col], plot_df[span_b_col], where=mask, alpha=0.12, interpolate=True)
    ax2.set_title("Ichimoku Cloud")
    ax2.legend(loc="upper left", ncol=3, fontsize=9)
    ax2.grid(True, alpha=0.2)

    # 3) MACD
    if macd_col and macd_col in plot_df.columns:
        ax3.plot(plot_df.index, plot_df[macd_col], label="MACD", linewidth=1.0)
    ax3.plot(plot_df.index, plot_df[macds_col], label="Signal", linewidth=1.0)
    ax3.bar(plot_df.index, plot_df[macdh_col], label="Hist", alpha=0.35)
    ax3.axhline(0, linewidth=0.8)
    ax3.set_title("MACD")
    ax3.legend(loc="upper left", ncol=3, fontsize=9)
    ax3.grid(True, alpha=0.2)

    # 4) RSI + MFI
    ax4.plot(plot_df.index, plot_df["RSI14"], label="RSI(14)", linewidth=1.0)
    ax4.plot(plot_df.index, plot_df["MFI14"], label="MFI(14)", linewidth=1.0)
    ax4.axhline(70, linewidth=0.8, alpha=0.7)
    ax4.axhline(30, linewidth=0.8, alpha=0.7)
    ax4.set_ylim(0, 100)
    ax4.set_title("RSI / MFI")
    ax4.legend(loc="upper left", ncol=2, fontsize=9)
    ax4.grid(True, alpha=0.2)

    # 5) OBV
    ax5.plot(plot_df.index, plot_df["OBV"], label="OBV", linewidth=1.0)
    ax5.plot(plot_df.index, plot_df["OBV_MA20"], label="OBV MA20", linewidth=1.0)
    ax5.set_title("OBV")
    ax5.legend(loc="upper left", ncol=2, fontsize=9)
    ax5.grid(True, alpha=0.2)

    # 6) SCORE + ADX + thresholds + regime shading
    ax6.plot(plot_df.index, plot_df["SCORE_100"], label="SCORE(0-100)", linewidth=1.2)
    ax6.plot(plot_df.index, plot_df["ADX14"], label="ADX(14)", linewidth=1.0, alpha=0.9)
    ax6.axhline(cfg.score_threshold, linewidth=0.9, alpha=0.8)
    ax6.axhline(cfg.adx_threshold, linewidth=0.9, alpha=0.5)
    ax6.set_ylim(0, 100)
    ax6.set_title("Score / ADX (Regime)")
    ax6.legend(loc="upper left", ncol=3, fontsize=9)
    ax6.grid(True, alpha=0.2)

    # 레짐 음영(추세장)
    trend_mask = plot_df["REGIME"] == "TREND"
    # fill_between으로 배경 음영 처리
    ax6.fill_between(plot_df.index, 0, 100, where=trend_mask, alpha=0.06, interpolate=True)

    plt.tight_layout()

    if os.environ.get("DISPLAY", "") == "":
        out = f"{ticker.replace('.', '_')}_analysis_plus.png"
        plt.savefig(out, dpi=150)
        print(f"[INFO] 화면 표시 불가 환경 → 차트를 '{out}'로 저장했습니다.")
    else:
        plt.show()


# -----------------------------
# Main
# -----------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="7대 지표 기반 주식 종합 분석기 + 레짐 + 백테스트")
    parser.add_argument("--ticker", type=str, required=True, help="예: 005930.KS")
    parser.add_argument("--daily_period", type=str, default="7y", help="일봉 데이터 기간(기본 7y)")
    parser.add_argument("--weekly_period", type=str, default="10y", help="주봉 데이터 기간(기본 10y)")
    parser.add_argument("--plot_bars", type=int, default=250, help="차트 표시 최근 봉 수(기본 250)")

    # Config overrides
    parser.add_argument("--adx_threshold", type=float, default=25.0)
    parser.add_argument("--trend_boost", type=float, default=1.15)
    parser.add_argument("--meanrev_boost", type=float, default=1.15)
    parser.add_argument("--atr_pct_threshold", type=float, default=0.05)
    parser.add_argument("--score_threshold", type=float, default=70.0)
    parser.add_argument("--horizon_days", type=int, default=5)
    parser.add_argument("--no_backtest", action="store_true", help="백테스트 출력 생략")

    args = parser.parse_args()

    cfg = ScoreConfig(
        adx_threshold=float(args.adx_threshold),
        trend_boost=float(args.trend_boost),
        meanrev_boost=float(args.meanrev_boost),
        # 반대 계열은 1/boost가 아니라 기존 설계대로 0.85로 유지하되, boost 변화에 맞춰 자동 조정
        # (원하면 여기서 같이 옵션화 가능)
        meanrev_cut=0.85,
        trend_cut=0.85,
        atr_pct_threshold=float(args.atr_pct_threshold),
        score_threshold=float(args.score_threshold),
        horizon_days=int(args.horizon_days),
    )

    ticker = args.ticker.strip()

    # 1) 데이터 수집: 일봉 + 주봉
    daily = download_ohlcv(ticker, interval="1d", period=args.daily_period)
    weekly = download_ohlcv(ticker, interval="1wk", period=args.weekly_period)

    # 2) 지표 계산
    daily_i = add_indicators(daily)
    weekly_i = add_indicators(weekly)

    # 3) 점수 시계열 생성(일봉/주봉)
    daily_scored = compute_score_series(daily_i, cfg)
    weekly_scored = compute_score_series(weekly_i, cfg)

    # 4) 최신 리포트
    last_date = daily_scored.index[-1].date()
    last_close = float(daily_scored["Close"].iloc[-1])
    latest = latest_score_detail(daily_scored, cfg)

    print("\n" + "=" * 78)
    print(f"[티커] {ticker}")
    print(f"[기준일] {last_date} (일봉 종가: {last_close:,.0f})")
    print(f"[레짐] {latest.details['regime']}  |  [등급] {latest.grade}")
    print(f"[종합 점수] {latest.score_100:.1f} / 100   (Raw: {latest.raw_points:+.1f})")
    print("-" * 78)

    print("[핵심 수치]")
    print(f" - Close: {latest.details['close']}")
    print(f" - VWAP:  {latest.details['vwap']}")
    print(f" - CMF20: {latest.details['cmf20']}")
    print(f" - ATR%:  {latest.details['atr_pct']}")

    print("-" * 78)
    print("[신호 요약]")
    for k, v in latest.details["signals"].items():
        print(f" - {k}: {v}")

    print("-" * 78)
    print("[일봉 지표 요약]")
    for line in summarize_indicators(daily_scored):
        print(f" - {line}")

    print("-" * 78)
    print("[주봉 추세 요약(참고)]")
    for line in summarize_indicators(weekly_scored)[:4]:
        print(f" - {line}")

    # 5) 간단 백테스트
    if not args.no_backtest:
        bt = simple_backtest(daily_scored, cfg)
        print("-" * 78)
        print(f"[간단 백테스트] 신호: SCORE >= {cfg.score_threshold:.1f}  |  기간: 다음날~{cfg.horizon_days}일")
        # 보기 좋게 출력
        with pd.option_context("display.max_rows", 200, "display.max_columns", 50, "display.width", 120):
            print(bt.to_string(index=False))
        print("=" * 78 + "\n")
    else:
        print("=" * 78 + "\n")

    # 6) 차트 출력(일봉 기준)
    plot_indicators_and_score(daily_scored, ticker=ticker, cfg=cfg, bars=args.plot_bars)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        raise SystemExit(1)
