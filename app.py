import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from config import NIFTY50

st.set_page_config(page_title="ManasAI Pro", page_icon="📈", layout="wide")

FEATURES = [
    "rsi14","macd_hist","ema20_gap","ema50_gap","ema200_gap",
    "atr_pct","vol_ratio","roc20","roc60","adx14","bb_pos",
    "drawdown_252","rel_strength_60"
]

@st.cache_data(ttl=1800, show_spinner=False)
def download_prices(tickers, period="10y"):
    frames = {}
    for t in tickers:
        try:
            df = yf.download(t, period=period, auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.rename(columns=str.lower)
            if "close" in df and len(df) > 250:
                frames[t] = df[["open","high","low","close","volume"]].dropna()
        except Exception:
            pass
    return frames

def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100/(1+rs)

def atr(df, n=14):
    prev = df.close.shift(1)
    tr = pd.concat([
        df.high-df.low, (df.high-prev).abs(), (df.low-prev).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def adx(df, n=14):
    up = df.high.diff()
    dn = -df.low.diff()
    plus = np.where((up > dn) & (up > 0), up, 0.0)
    minus = np.where((dn > up) & (dn > 0), dn, 0.0)
    a = atr(df, n).replace(0, np.nan)
    pdi = 100 * pd.Series(plus, index=df.index).ewm(alpha=1/n, adjust=False).mean()/a
    mdi = 100 * pd.Series(minus, index=df.index).ewm(alpha=1/n, adjust=False).mean()/a
    dx = (100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan))
    return dx.ewm(alpha=1/n, adjust=False).mean()

def features(df):
    x = df.copy()
    c = x.close
    e20,e50,e200 = ema(c,20),ema(c,50),ema(c,200)
    mid = c.rolling(20).mean()
    sd = c.rolling(20).std()
    x["rsi14"] = rsi(c)
    macd = ema(c,12)-ema(c,26)
    x["macd_hist"] = macd-ema(macd,9)
    x["ema20_gap"] = c/e20-1
    x["ema50_gap"] = c/e50-1
    x["ema200_gap"] = c/e200-1
    x["atr_pct"] = atr(x)/c
    x["vol_ratio"] = x.volume/x.volume.rolling(20).mean()
    x["roc20"] = c.pct_change(20)
    x["roc60"] = c.pct_change(60)
    x["adx14"] = adx(x)
    x["bb_pos"] = (c-(mid-2*sd))/(4*sd)
    x["drawdown_252"] = c/c.rolling(252).max()-1
    x["rel_strength_60"] = c.pct_change(60)
    return x.replace([np.inf,-np.inf],np.nan)

def rule_score(row):
    score = 50.0
    # trend 25
    score += np.clip(row.ema20_gap*250, -12, 12)
    score += np.clip(row.ema50_gap*200, -10, 10)
    score += np.clip(row.ema200_gap*100, -8, 8)
    # momentum 25
    score += np.clip((row.rsi14-50)*0.22, -8, 8)
    score += np.clip(row.roc20*100*0.12, -6, 6)
    score += np.clip(row.roc60*100*0.10, -6, 6)
    score += np.clip(row.macd_hist/row.close*10000, -5, 5)
    # volume/quality 15
    score += np.clip((row.vol_ratio-1)*4, -4, 4)
    score += np.clip((row.adx14-20)*0.15, -3, 3)
    # risk penalty 10
    score += np.clip(row.drawdown_252*10, -8, 0)
    score -= np.clip(row.atr_pct*100*0.8, 0, 5)
    return float(np.clip(score,0,100))

def build_dataset(prices):
    rows=[]
    for ticker,df in prices.items():
        f=features(df)
        f["ticker"]=ticker
        # 20-trading-day forward return target
        f["fwd20"]=df.close.shift(-20)/df.close-1
        f["target"]=(f.fwd20>0.03).astype(int)
        rows.append(f.reset_index())
    return pd.concat(rows,ignore_index=True).dropna(subset=FEATURES+["target"])

def train_model(dataset):
    # Time split avoids random leakage.
    dataset=dataset.sort_values("Date")
    cut=int(len(dataset)*0.80)
    train,test=dataset.iloc[:cut],dataset.iloc[cut:]
    model=RandomForestClassifier(
        n_estimators=300,max_depth=7,min_samples_leaf=20,
        class_weight="balanced",random_state=42,n_jobs=-1
    )
    model.fit(train[FEATURES],train.target)
    prob=model.predict_proba(test[FEATURES])[:,1]
    pred=(prob>=0.5).astype(int)
    auc=roc_auc_score(test.target,prob) if test.target.nunique()>1 else np.nan
    return model, auc, accuracy_score(test.target,pred)

def technical_report(ticker,df):
    f=features(df).dropna()
    r=f.iloc[-1]
    c=float(r.close)
    recent=df.tail(120)
    # simple pivot-style levels
    support=float(recent.low.tail(40).quantile(0.20))
    resistance=float(recent.high.tail(40).quantile(0.80))
    score=rule_score(r)
    trend="Bullish" if r.ema20_gap>0 and r.ema50_gap>0 and r.ema200_gap>0 else ("Bearish" if r.ema20_gap<0 and r.ema50_gap<0 else "Mixed")
    momentum="Strong" if r.rsi14>=60 and r.macd_hist>0 else ("Weak" if r.rsi14<45 and r.macd_hist<0 else "Neutral")
    risk="High" if r.atr_pct>0.04 else ("Moderate" if r.atr_pct>0.025 else "Lower")
    return {
        "price":c,"score":score,"trend":trend,"momentum":momentum,"risk":risk,
        "rsi":float(r.rsi14),"adx":float(r.adx14),"vol_ratio":float(r.vol_ratio),
        "support":support,"resistance":resistance
    }

st.title("📈 ManasAI Pro")
st.caption("Educational quantitative research engine — InvestingPro-style, but independently built.")

with st.sidebar:
    st.header("Universe")
    universe=st.multiselect("Stocks",NIFTY50,default=NIFTY50[:20])
    period=st.selectbox("History",["5y","10y"],index=1)
    topn=st.slider("Top N",5,20,10)
    st.info("This prototype is for education. Do not use scores as guaranteed forecasts.")

if not universe:
    st.warning("Select at least one stock.")
    st.stop()

with st.spinner("Downloading market data and building features..."):
    prices=download_prices(tuple(universe),period)

if len(prices)<3:
    st.error("Not enough data returned. Check internet access or Yahoo Finance availability.")
    st.stop()

dataset=build_dataset(prices)
model,auc,acc=train_model(dataset)

latest=[]
for ticker,df in prices.items():
    f=features(df).dropna()
    r=f.iloc[-1]
    p=float(model.predict_proba(pd.DataFrame([r[FEATURES].values],columns=FEATURES))[0,1])
    q=rule_score(r)
    final=0.60*q+0.40*(p*100)
    latest.append([ticker,float(r.close),q,p*100,final,float(r.rsi14),float(r.roc20)])
rank=pd.DataFrame(latest,columns=["Ticker","Price","RuleScore","MLProbability","AI Score","RSI","20D Return"])
rank=rank.sort_values("AI Score",ascending=False)

c1,c2,c3,c4=st.columns(4)
c1.metric("Stocks analyzed",len(rank))
c2.metric("Top AI score",f"{rank['AI Score'].max():.1f}")
c3.metric("Model ROC-AUC",f"{auc:.2f}" if pd.notna(auc) else "N/A")
c4.metric("Model accuracy",f"{acc:.1%}")

st.subheader("🏆 AI-ranked stocks")
st.dataframe(
    rank.style.format({"Price":"₹{:.2f}","RuleScore":"{:.1f}","MLProbability":"{:.1f}","AI Score":"{:.1f}","RSI":"{:.1f}","20D Return":"{:.1%}"}),
    use_container_width=True,hide_index=True
)

selected=st.selectbox("Analyze stock",rank.Ticker.tolist(),index=0)
df=prices[selected]
rep=technical_report(selected,df)

left,right=st.columns([2,1])
with left:
    st.subheader(f"{selected} — AI Chart Analysis")
    fig=go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index[-250:],open=df.open[-250:],high=df.high[-250:],
        low=df.low[-250:],close=df.close[-250:],name="Price"))
    f=features(df)
    fig.add_trace(go.Scatter(x=df.index[-250:],y=ema(df.close,20)[-250:],name="EMA20"))
    fig.add_trace(go.Scatter(x=df.index[-250:],y=ema(df.close,50)[-250:],name="EMA50"))
    fig.add_trace(go.Scatter(x=df.index[-250:],y=ema(df.close,200)[-250:],name="EMA200"))
    fig.update_layout(height=520,xaxis_rangeslider_visible=False)
    st.plotly_chart(fig,use_container_width=True)

with right:
    st.metric("AI Score",f"{rep['score']:.1f}/100")
    st.write(f"**Trend:** {rep['trend']}")
    st.write(f"**Momentum:** {rep['momentum']}")
    st.write(f"**Risk:** {rep['risk']}")
    st.write(f"**RSI:** {rep['rsi']:.1f}")
    st.write(f"**ADX:** {rep['adx']:.1f}")
    st.write(f"**Volume:** {rep['vol_ratio']:.2f}× 20D average")
    st.write(f"**Support:** ₹{rep['support']:.2f}")
    st.write(f"**Resistance:** ₹{rep['resistance']:.2f}")

st.subheader("🧠 Why the model likes/dislikes it")
reasons=[]
if rep["trend"]=="Bullish": reasons.append("Price is above the main EMA trend structure.")
if rep["trend"]=="Bearish": reasons.append("Price is below the main EMA trend structure.")
if rep["rsi"]>=60: reasons.append("Momentum is positive, with RSI above 60.")
elif rep["rsi"]<45: reasons.append("Momentum is weak, with RSI below 45.")
if rep["vol_ratio"]>=1.3: reasons.append("Trading volume is materially above its 20-day average.")
if rep["adx"]>=25: reasons.append("ADX indicates a relatively strong trend.")
if rep["risk"]=="High": reasons.append("ATR-based volatility is high, increasing position risk.")
if not reasons: reasons.append("Signals are mixed; the model has limited conviction.")
for x in reasons: st.write("•",x)

st.caption("Research prototype only. The ML probability is historical-model output, not a guaranteed probability of future returns.")
