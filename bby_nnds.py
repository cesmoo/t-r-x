"""
====================================================================================
🏆 THE ULTIMATE META-AI SUPER ENSEMBLE (PRO-10 CORE) 🏆
====================================================================================
Architecture: Asynchronous Producer-Consumer Pattern
Engines (10 Cores): RF, GB, Bayes, LSTM, Markov, NGram, MonteCarlo, Trend, Entropy, Meta-Trainer
Risk System: Kelly Criterion Probability Assessor
Resilience: Exponential Backoff, Connection Pooling, Flood Control
====================================================================================
"""

import asyncio
import time
import os
import logging
import random
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from typing import List, Tuple, Dict, Any, Optional

import aiohttp
from aiohttp import ClientTimeout, TCPConnector
import motor.motor_asyncio 

from aiogram import Bot, Dispatcher
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties
from aiogram.exceptions import TelegramRetryAfter
from dotenv import load_dotenv

import numpy as np
import scipy.stats as stats
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim

import warnings
warnings.filterwarnings("ignore")

# =========================================================================
# ⚙️ MODULE 1: ENTERPRISE CONFIG & LOGGING
# =========================================================================
load_dotenv()
MMT = timezone(timedelta(hours=6, minutes=30))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(module)s:%(funcName)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("ULTRA-PRO")

class Config:
    BOT_TOKEN = os.getenv("BOT_TOKEN", "YOUR_BOT_TOKEN")
    CHANNEL_ID = os.getenv("CHANNEL_ID", "YOUR_CHANNEL_ID")
    MONGO_URI = os.getenv("MONGO_URI", "")
    
    API_URL = 'https://api.bigwinqaz.com/api/webapi/GetNoaverageEmerdList'
    REQUEST_TIMEOUT = 10.0
    
    WIN_STICKER = "CAACAgUAAxkBAAEQwtVpt1_oWxyaQFmiy3O_1knZjN9yCwAC2hIAAikFkVX0qhu40v6REDoE"
    LOSE_STICKER = ""
    
    @staticmethod
    def get_headers() -> Dict[str, str]:
        return {
            'authority': 'api.bigwinqaz.com', 
            'accept': 'application/json, text/plain, */*',
            'content-type': 'application/json;charset=UTF-8', 
            'origin': 'https://www.777bigwingame.app',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

bot = Bot(token=Config.BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()

# =========================================================================
# 📦 MODULE 2: DATA MODELS (Strict Typing)
# =========================================================================
@dataclass
class GameRecord:
    issue_number: str
    number: int
    size: str
    parity: str
    timestamp: datetime

# =========================================================================
# 🗄️ MODULE 3: ROBUST DATABASE MANAGER
# =========================================================================
class DatabaseManager:
    def __init__(self, uri: str):
        self.client = motor.motor_asyncio.AsyncIOMotorClient(
            uri, maxPoolSize=100, minPoolSize=10, serverSelectionTimeoutMS=5000
        )
        self.db = self.client['bigwin4pattern_database']
        self.history = self.db['game_history']
        self.predictions = self.db['predictions']

    async def initialize(self) -> None:
        await self.history.create_index("issue_number", unique=True)
        await self.predictions.create_index("issue_number", unique=True)
        logger.info("✅ Database Connection Pool Established.")

    async def save_record(self, record: GameRecord) -> bool:
        try:
            res = await self.history.update_one(
                {"issue_number": record.issue_number}, 
                {"$setOnInsert": asdict(record)}, 
                upsert=True
            )
            return res.upserted_id is not None
        except Exception as e:
            logger.error(f"DB Write Error: {e}")
            return False

    async def save_prediction(self, issue: str, pred: str, prob: float, conf: float, kelly: float, status: str, details: Dict[str, float]):
        try:
            doc = {
                "predicted_size": pred, "win_probability": prob, "confidence": conf, 
                "kelly_fraction": kelly, "status": status, "model_votes": details, 
                "timestamp": datetime.now(MMT)
            }
            await self.predictions.update_one({"issue_number": issue}, {"$set": doc}, upsert=True)
        except Exception as e:
            logger.error(f"Save Prediction Error: {e}")

    async def update_result(self, issue: str, actual_size: str, is_win: bool):
        try:
            await self.predictions.update_one(
                {"issue_number": issue},
                {"$set": {"actual_size": actual_size, "is_win": is_win}}
            )
        except Exception as e:
            logger.error(f"Update Result Error: {e}")

    async def get_history(self, limit: int = 500) -> List[Dict[str, Any]]:
        return await self.history.find().sort("issue_number", -1).limit(limit).to_list(length=limit)

# =========================================================================
# 🔬 MODULE 4: FEATURE ENGINEERING
# =========================================================================
class FeatureEngineer:
    def __init__(self, window_size: int = 6):
        self.window = window_size
        self.scaler = StandardScaler()

    def extract_features(self, sizes: List[str], numbers: List[int], parities: List[str]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        if len(sizes) < self.window * 4: return None, None, None
        X, y = [], []
        try:
            for i in range(len(sizes) - self.window):
                row = []
                for j in range(self.window):
                    size_val = 1.0 if sizes[i+j] == 'BIG' else 0.0
                    par_val = 1.0 if parities[i+j] == 'EVEN' else 0.0
                    row.extend([size_val, par_val, float(numbers[i+j])])
                X.append(row)
                y.append(1.0 if sizes[i+self.window] == 'BIG' else 0.0)
            
            curr_feats = []
            for j in range(1, self.window + 1):
                size_val = 1.0 if sizes[-j] == 'BIG' else 0.0
                par_val = 1.0 if parities[-j] == 'EVEN' else 0.0
                curr_feats.extend([size_val, par_val, float(numbers[-j])])
                
            X_scaled = self.scaler.fit_transform(X)
            curr_scaled = self.scaler.transform([curr_feats])
            return X_scaled, np.array(y), curr_scaled
        except Exception as e:
            logger.error(f"Feature Engineering Error: {e}")
            return None, None, None

# =========================================================================
# 🧠 MODULE 5: THE 10 AI CORES (PREDICTIVE ENGINES)
# =========================================================================
class TreeEngines:
    def __init__(self):
        self.rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        self.gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
    def predict(self, X: np.ndarray, y: np.ndarray, curr_X: np.ndarray) -> Tuple[float, float]:
        try:
            self.rf.fit(X, y)
            self.gb.fit(X, y)
            rf_prob = float(self.rf.predict_proba(curr_X)[0][1]) if 1.0 in self.rf.classes_ else 0.5
            gb_prob = float(self.gb.predict_proba(curr_X)[0][1]) if 1.0 in self.gb.classes_ else 0.5
            return rf_prob, gb_prob
        except: return 0.5, 0.5

class BayesianEngine:
    def __init__(self): self.nb = GaussianNB()
    def predict(self, X: np.ndarray, y: np.ndarray, curr_X: np.ndarray) -> float:
        try:
            self.nb.fit(X, y)
            return float(self.nb.predict_proba(curr_X)[0][1]) if 1.0 in self.nb.classes_ else 0.5
        except: return 0.5

class SimpleLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=16, num_layers=1, batch_first=True)
        self.fc = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.sigmoid(self.fc(hn[-1]))

class LSTMEngine:
    def predict(self, sizes: List[str]) -> float:
        if len(sizes) < 50: return 0.5
        try:
            data = [1.0 if s == 'BIG' else 0.0 for s in sizes[-50:]]
            X_t = torch.tensor(data[:-1], dtype=torch.float32).view(1, -1, 1)
            y_t = torch.tensor([data[-1]], dtype=torch.float32).view(1, 1)
            
            model = SimpleLSTM()
            optimizer = optim.Adam(model.parameters(), lr=0.01)
            criterion = nn.BCELoss()
            
            model.train()
            for _ in range(5): 
                optimizer.zero_grad()
                loss = criterion(model(X_t), y_t)
                loss.backward()
                optimizer.step()
            
            model.eval()
            curr_X = torch.tensor(data[1:], dtype=torch.float32).view(1, -1, 1)
            with torch.no_grad(): return float(model(curr_X).item())
        except: return 0.5

class MarkovEngine:
    @staticmethod
    def predict(sizes: List[str]) -> float:
        if len(sizes) < 10: return 0.5
        trans = {'BIG': {'BIG': 0, 'SMALL': 0}, 'SMALL': {'BIG': 0, 'SMALL': 0}}
        try:
            for i in range(len(sizes)-1): trans[sizes[i]][sizes[i+1]] += 1
            curr = sizes[-1]
            tot = sum(trans[curr].values())
            return float(trans[curr]['BIG'] / tot) if tot > 0 else 0.5
        except: return 0.5

class NGramEngine:
    @staticmethod
    def predict(sizes: List[str], n: int = 4) -> float:
        if len(sizes) < n+1: return 0.5
        try:
            pat = tuple(sizes[-n:])
            matches = [sizes[i+n] for i in range(len(sizes)-n) if tuple(sizes[i:i+n]) == pat]
            return float(matches.count('BIG') / len(matches)) if matches else 0.5
        except: return 0.5

class MonteCarloEngine:
    @staticmethod
    def predict(sizes: List[str], sims: int = 1000) -> float:
        if not sizes: return 0.5
        try:
            prob_b = sizes.count('BIG') / len(sizes)
            results = np.random.choice([1.0, 0.0], size=sims, p=[prob_b, 1.0-prob_b])
            return float(np.mean(results))
        except: return 0.5

class TrendEngine:
    @staticmethod
    def predict(sizes: List[str], window: int = 15) -> float:
        if len(sizes) < window: return 0.5
        try:
            momentum = sizes[-window:].count('BIG') / float(window)
            if momentum >= 0.8: return 0.25 
            if momentum <= 0.2: return 0.75 
            return float(momentum)
        except: return 0.5

class EntropyEngine:
    @staticmethod
    def predict(sizes: List[str]) -> float:
        if len(sizes) < 20: return 0.5
        try:
            p_b = sizes[-20:].count('BIG') / 20.0
            p_s = 1.0 - p_b
            if p_b == 0 or p_s == 0: return float(p_b)
            entropy = stats.entropy([p_b, p_s], base=2)
            if entropy > 0.98: return 0.5 
            return float(p_b)
        except: return 0.5

# =========================================================================
# 🧬 MODULE 6: THE META-OPTIMIZER & RISK MANAGER
# =========================================================================
class ReinforcementTrainer:
    """ Core 10: Dynamic Weight Adjustment (Self-Learning Hub) """
    def __init__(self):
        self.weights: Dict[str, float] = {
            'rf': 0.15, 'gb': 0.15, 'bayes': 0.10, 'lstm': 0.15,
            'markov': 0.10, 'ngram': 0.10, 'monte': 0.05, 'trend': 0.10, 'entropy': 0.10
        }

    def update(self, actual_size: str, past_preds: Dict[str, float]) -> None:
        if not past_preds: return
        try:
            actual_val = 1.0 if actual_size == 'BIG' else 0.0
            total_w = 0.0
            for model, prob in past_preds.items():
                error = abs(actual_val - prob)
                if error < 0.4: self.weights[model] += 0.05
                else: self.weights[model] = max(0.01, self.weights[model] - 0.02)
                total_w += self.weights[model]
                
            if total_w > 0:
                for k in self.weights: self.weights[k] = float(self.weights[k] / total_w)
        except Exception as e: logger.error(f"Trainer Error: {e}")

class KellyRiskManager:
    """ Probability Theory (Kelly Formula) ကိုသုံး၍ Risk တွက်ချက်ခြင်း """
    def __init__(self, win_probability_threshold: float = 0.54):
        self.threshold = win_probability_threshold

    def calculate_kelly_fraction(self, win_prob: float, odds: float = 1.95) -> float:
        if win_prob < self.threshold: return 0.0
        b = odds - 1.0
        f_star = (b * win_prob - (1.0 - win_prob)) / b
        return max(0.0, round(f_star, 4))

# =========================================================================
# ⚙️ MODULE 7: SUPER ENSEMBLE ORCHESTRATOR
# =========================================================================
class MetaSuperEngine:
    def __init__(self):
        self.fe = FeatureEngineer()
        self.opt = ReinforcementTrainer()
        self.risk = KellyRiskManager()
        self.trees = TreeEngines()
        self.bayes = BayesianEngine()
        self.lstm = LSTMEngine()
        self.last_probs: Dict[str, float] = {}

    async def analyze(self, docs: List[Dict[str, Any]]) -> Tuple[float, str, float, float, str, Dict[str, float]]:
        if len(docs) < 50: return 0.5, random.choice(["BIG", "SMALL"]), 50.0, 0.0, "SKIPPED", {}
        
        try:
            sizes = [d.get('size', 'BIG') for d in reversed(docs)]
            nums = [int(d.get('number', 0)) for d in reversed(docs)]
            pars = [d.get('parity', 'EVEN') for d in reversed(docs)]
            baseline_b = sizes.count('BIG') / len(sizes)
            if baseline_b in [0, 1]: baseline_b = 0.5 

            X, y, curr_X = self.fe.extract_features(sizes, nums, pars)
            
            # 🚀 META-AI: Concurrently run all models to prevent Bot Freezing
            async def run_trees(): return await asyncio.to_thread(self.trees.predict, X, y, curr_X) if X is not None and len(X) > 10 else (baseline_b, baseline_b)
            async def run_bayes(): return await asyncio.to_thread(self.bayes.predict, X, y, curr_X) if X is not None and len(X) > 10 else baseline_b

            tasks = [
                run_trees(), run_bayes(),
                asyncio.to_thread(self.lstm.predict, sizes),
                asyncio.to_thread(MarkovEngine.predict, sizes),
                asyncio.to_thread(NGramEngine.predict, sizes),
                asyncio.to_thread(MonteCarloEngine.predict, sizes),
                asyncio.to_thread(TrendEngine.predict, sizes),
                asyncio.to_thread(EntropyEngine.predict, sizes)
            ]
            results = await asyncio.gather(*tasks)
            (rf_p, gb_p), bayes_p, lstm_p, markov_p, ngram_p, monte_p, trend_p, entropy_p = results
            
            probs = {
                'rf': rf_p, 'gb': gb_p, 'bayes': bayes_p, 'lstm': lstm_p,
                'markov': markov_p, 'ngram': ngram_p, 'monte': monte_p,
                'trend': trend_p, 'entropy': entropy_p
            }
            self.last_probs = {k: float(v) for k, v in probs.items()}
            
            # Blending with Reinforcement Weights
            w = self.opt.weights
            final_b = sum(probs[k] * w.get(k, 0.1) for k in probs)
            
            pred_size = "BIG" if final_b > baseline_b else "SMALL"
            
            # Confidence & Kelly Calculation
            win_prob = max(final_b, 1.0 - final_b) 
            conf_percent = min(max((0.5 + abs(final_b - baseline_b) * 2.5) * 100, 50.1), 99.0)
            kelly_fraction = self.risk.calculate_kelly_fraction(win_prob)
            
            status = "APPROVED" if kelly_fraction > 0 else "SKIPPED (High Risk)"
            
            return win_prob, pred_size, round(conf_percent, 1), kelly_fraction, status, self.last_probs
            
        except Exception as e:
            logger.error(f"Meta Engine Error: {e}")
            return 0.5, random.choice(["BIG", "SMALL"]), 50.0, 0.0, "SKIPPED", {}

# =========================================================================
# 🚀 MODULE 8: PRODUCER-CONSUMER MAIN CONTROLLER
# =========================================================================
class EnterpriseController:
    def __init__(self):
        self.db = DatabaseManager(Config.MONGO_URI)
        self.ai = MetaSuperEngine()
        self.task_queue = asyncio.Queue(maxsize=100)
        self.last_issue = None

    async def api_producer_loop(self):
        """ Data ဆွဲယူပေးမည့်အပိုင်း (Auto Backoff & Connection Pooling ပါဝင်သည်) """
        timeout = ClientTimeout(total=Config.REQUEST_TIMEOUT)
        connector = TCPConnector(limit=10, keepalive_timeout=60)
        
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            logger.info("📡 API Producer Service Started.")
            backoff = 1.0
            
            while True:
                try:
                    payload = {'pageSize': 5, 'pageNo': 1, 'typeId': 30, 'language': 7, 'random': 'c67645ba506f4653a98639179e216677', 'signature': 'CEB3DF3A115BDC3551A20AC8842A6A85', 'timestamp': int(time.time())}
                    async with session.post(Config.API_URL, headers=Config.get_headers(), json=payload) as r:
                        if r.status == 200:
                            data = await r.json()
                            if data.get('code') == 0:
                                latest = data["data"]["list"][0]
                                issue = str(latest["issueNumber"])
                                
                                if issue != self.last_issue:
                                    record = GameRecord(
                                        issue_number=issue, number=int(latest["number"]),
                                        size="BIG" if int(latest["number"]) >= 5 else "SMALL",
                                        parity="EVEN" if int(latest["number"]) % 2 == 0 else "ODD",
                                        timestamp=datetime.now(MMT)
                                    )
                                    await self.task_queue.put(record)
                                    self.last_issue = issue
                                    backoff = 1.0
                        else: raise Exception(f"HTTP {r.status}")
                except Exception as e:
                    logger.warning(f"⚠️ API Fetch Failed: {e}. Retrying in {backoff}s")
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 10.0)
                
                await asyncio.sleep(1.5)

    async def ml_consumer_loop(self):
        """ Queue မှ Data ရယူပြီး AI Model တွက်ချက်မည့်အပိုင်း """
        logger.info("🧠 ML Consumer Worker Started.")
        while True:
            new_record: GameRecord = await self.task_queue.get()
            logger.info(f"🟢 Processing New Issue: {new_record.issue_number} -> {new_record.size}")
            
            # 1. Self-Learning (Update Meta Weights)
            self.ai.opt.update(new_record.size, self.ai.last_probs)

            # 2. Check Previous Prediction Result & Send UI
            pred_doc = await self.db.predictions.find_one({"issue_number": new_record.issue_number})
            if pred_doc and pred_doc.get('status') == "APPROVED":
                pred_size = str(pred_doc['predicted_size'])
                is_win = (pred_size == new_record.size)
                await self.db.update_result(new_record.issue_number, new_record.size, is_win)
                await self._broadcast_result(new_record.issue_number, pred_size, is_win, new_record.size, new_record.number)

            # 3. Save History
            is_new = await self.db.save_record(new_record)
            if not is_new: 
                self.task_queue.task_done()
                continue

            # 4. Predict Next Issue
            history = await self.db.get_history(500)
            next_issue = str(int(new_record.issue_number) + 1)
            
            logger.info(f"⏳ Async Parallel Analyzing for {next_issue}...")
            win_prob, pred_size, conf, kelly_f, status, details = await self.ai.analyze(history)
            
            await self.db.save_prediction(next_issue, pred_size, win_prob, conf, kelly_f, status, details)
            
            top_model = max(self.ai.opt.weights, key=self.ai.opt.weights.get) if self.ai.opt.weights else "N/A"
            await self._broadcast_prediction(next_issue, pred_size, conf, kelly_f, status, top_model)
            
            self.task_queue.task_done()

    async def _broadcast_prediction(self, issue: str, pred: str, conf: float, kelly: float, status: str, top_engine: str):
        kelly_pct = round(kelly * 100, 2)
        if status == "SKIPPED (High Risk)":
            msg = (f"<b>[META-AI SUPER ENSEMBLE]</b>\n⏰ Period: <code>{issue}</code>\n"
                   f"⚠️ Prediction: <b>SKIPPED</b>\n📊 Confidence: {conf}% (Win Prob Too Low)\n"
                   f"🧠 Top Engine: <code>{top_engine.upper()}</code>")
        else:
            msg = (f"<b>[META-AI SUPER ENSEMBLE]</b>\n⏰ Period: <code>{issue}</code>\n"
                   f"🎯 Prediction: <b>{pred}</b>\n📊 Confidence: {conf}%\n"
                   f"💼 Kelly Bet Size: <b>{kelly_pct}%</b> of Bankroll\n"
                   f"🧠 Top Engine: <code>{top_engine.upper()}</code>")
        try:
            await bot.send_message(chat_id=Config.CHANNEL_ID, text=msg)
        except TelegramRetryAfter as e:
            await asyncio.sleep(e.retry_after)
            await bot.send_message(chat_id=Config.CHANNEL_ID, text=msg)
        except Exception as e: logger.error(f"UI Predict Error: {e}")

    async def _broadcast_result(self, issue: str, pred: str, is_win: bool, actual_size: str, actual_num: int):
        win_str, icon = ("WIN ✅", "🟢") if is_win else ("LOSE ❌", "🔴")
        res_letter = "B" if actual_size == "BIG" else "S"
        msg = (f"<b>🏆 META-AI RESULTS</b>\n\n⏰ Period: <code>{issue}</code>\n🎯 Choice: {pred}\n"
               f"📊 Result: {icon} <b>{win_str}</b> | {res_letter} ({actual_num})")
        try: 
            await bot.send_message(chat_id=Config.CHANNEL_ID, text=msg)
            if is_win and Config.WIN_STICKER: await bot.send_sticker(chat_id=Config.CHANNEL_ID, sticker=Config.WIN_STICKER)
        except Exception as e: logger.error(f"UI Result Error: {e}")

# =========================================================================
# 🚀 MODULE 9: GRACEFUL ENTRY POINT
# =========================================================================
async def main():
    logger.info("Initializing Enterprise AI System with 10 Cores...")
    await bot.delete_webhook(drop_pending_updates=True)
    
    controller = EnterpriseController()
    await controller.db.initialize()
    
    # Run API Fetcher and ML Engine concurrently
    producer_task = asyncio.create_task(controller.api_producer_loop())
    consumer_task = asyncio.create_task(controller.ml_consumer_loop())
    
    logger.info("System is Fully Operational.")
    try:
        await asyncio.gather(producer_task, consumer_task, dp.start_polling(bot))
    except asyncio.CancelledError:
        logger.info("System shutting down gracefully...")

if __name__ == '__main__':
    try: asyncio.run(main())
    except KeyboardInterrupt: logger.info("Application Stopped manually.")
