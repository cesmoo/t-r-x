"""
====================================================================================
🏆 ULTRA-AI COLOR PREDICTOR (10-CORE 4-CLASS ARCHITECTURE) 🏆
====================================================================================
Developer: Master AI System
Target: 6win566 Win Go (Color: RED, GREEN, RED_VIOLET, GREEN_VIOLET)
Architecture: Modular Object-Oriented Programming (OOP)
====================================================================================
"""

import asyncio
import time
import os
import logging
from collections import Counter
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional
from dotenv import load_dotenv

import aiohttp
import motor.motor_asyncio 

from aiogram import Bot, Dispatcher
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties

# --- 🧠 ULTRA AI & DATA SCIENCE LIBRARIES ---
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
# ⚙️ MODULE 1: SYSTEM CONFIGURATION
# =========================================================================
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("COLOR-AI-V2")

class Config:
    BOT_TOKEN: str = os.getenv("BOT_TOKEN", "YOUR_BOT_TOKEN")
    CHANNEL_ID: str = os.getenv("CHANNEL_ID", "YOUR_CHANNEL_ID")
    MONGO_URI: str = os.getenv("MONGO_URI", "YOUR_MONGO_URI")

    WIN_STICKER = "CAACAgUAAxkBAAEQwtVpt1_oWxyaQFmiy3O_1knZjN9yCwAC2hIAAikFkVX0qhu40v6REDoE"  
    LOSE_STICKER = "" 
    
    MULTIPLIERS: List[int] = [1, 2, 3, 5, 8, 15, 30, 50, 100]
    API_URL: str = 'https://6lotteryapi.com/api/webapi/GetNoaverageEmerdList'
    
    @staticmethod
    def get_headers() -> Dict[str, str]:
        """ API Request အတွက် Headers များကို ထုတ်ပေးသည် """
        return {
            'authority': '6lotteryapi.com', 
            'accept': 'application/json, text/plain, */*',
            # 💡 သတိပြုရန်: ဤနေရာတွင် Token အသစ်ကို အစားထိုးပါ
            'authorization': 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpYXQiOiIxNzczNjMxOTU4IiwibmJmIjoiMTc3MzYzMTk1OCIsImV4cCI6IjE3NzM2MzM3NTgiLCJodHRwOi8vc2NoZW1hcy5taWNyb3NvZnQuY29tL3dzLzIwMDgvMDYvaWRlbnRpdHkvY2xhaW1zL2V4cGlyYXRpb24iOiIzLzE2LzIwMjYgMTA6MzI6MzggQU0iLCJodHRwOi8vc2NoZW1hcy5taWNyb3NvZnQuY29tL3dzLzIwMDgvMDYvaWRlbnRpdHkvY2xhaW1zL3JvbGUiOiJBY2Nlc3NfVG9rZW4iLCJVc2VySWQiOiIxMDc2NTk0IiwiVXNlck5hbWUiOiI5NTk2NzUzMjM4NzgiLCJVc2VyUGhvdG8iOiI3IiwiTmlja05hbWUiOiLhgJXhgLzhgIrhgLfhgLrhgIXhgK_hgLYiLCJBbW91bnQiOiI5NS41NSIsIkludGVncmFsIjoiMCIsIkxvZ2luTWFyayI6Ikg1IiwiTG9naW5UaW1lIjoiMy8xNi8yMDI2IDEwOjAyOjM4IEFNIiwiTG9naW5JUEFkZHJlc3MiOiIxMDMuMTM0LjIwNy4xNTIiLCJEYk51bWJlciI6IjAiLCJJc3ZhbGlkYXRvciI6IjAiLCJLZXlDb2RlIjoiMTA2IiwiVG9rZW5UeXBlIjoiQWNjZXNzX1Rva2VuIiwiUGhvbmVUeXBlIjoiMSIsIlVzZXJUeXBlIjoiMCIsIlVzZXJOYW1lMiI6IiIsImlzcyI6Imp3dElzc3VlciIsImF1ZCI6ImxvdHRlcnlUaWNrZXQifQ.ODo3Y-5b0iaBUF8O05E1eOsxWj8WpKgIqETu0LeEWR8',
            'content-type': 'application/json;charset=UTF-8', 
            'origin': 'https://www.6win566.com',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

bot = Bot(token=Config.BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()

# =========================================================================
# 🗄️ MODULE 2: DATABASE MANAGER
# =========================================================================
class DatabaseManager:
    def __init__(self, uri: str):
        self.client = motor.motor_asyncio.AsyncIOMotorClient(uri, serverSelectionTimeoutMS=5000)
        self.db = self.client['sixlottery_color_ultra_v2'] 
        self.history = self.db['game_history']
        self.predictions = self.db['predictions']

    async def initialize(self) -> None:
        try:
            await self.history.create_index("issue_number", unique=True)
            await self.predictions.create_index("issue_number", unique=True)
            logger.info("✅ Database V2 Initialized Successfully.")
        except Exception as e: logger.error(f"❌ DB Error: {e}")

    async def save_history(self, issue: str, number: int, color: str) -> None:
        try:
            doc = {"number": int(number), "color": str(color), "timestamp": datetime.now()}
            await self.history.update_one({"issue_number": issue}, {"$setOnInsert": doc}, upsert=True)
        except Exception: pass

    async def save_prediction(self, issue: str, pred_color: str, confidence: float, details: Dict[str, Dict[str, float]]) -> None:
        try:
            doc = {"predicted_color": str(pred_color), "confidence": float(confidence), "model_votes": details, "timestamp": datetime.now()}
            await self.predictions.update_one({"issue_number": issue}, {"$set": doc}, upsert=True)
        except Exception: pass

    async def update_result(self, issue: str, actual_color: str, actual_number: int, win_lose: str) -> None:
        try:
            doc = {"actual_color": str(actual_color), "actual_number": int(actual_number), "win_lose": str(win_lose)}
            await self.predictions.update_one({"issue_number": issue}, {"$set": doc})
        except Exception: pass

    async def get_history(self, limit: int = 500) -> List[Dict[str, Any]]:
        return await self.history.find().sort("issue_number", -1).limit(limit).to_list(length=limit)

    async def get_recent_predictions(self, limit: int = 10) -> List[Dict[str, Any]]:
        return await self.predictions.find({"win_lose": {"$ne": None}}).sort("issue_number", -1).limit(limit).to_list(length=limit)

# =========================================================================
# 🔬 MODULE 3: 4-CLASS FEATURE ENGINEERING
# =========================================================================
# 💡 [NEW] Class ၄ မျိုး သတ်မှတ်ချက် (0: Red, 1: Green, 2: Red-Violet, 3: Green-Violet)
COLOR_MAP = {'RED': 0, 'GREEN': 1, 'RED_VIOLET': 2, 'GREEN_VIOLET': 3}
STATES = ['RED', 'GREEN', 'RED_VIOLET', 'GREEN_VIOLET']

def get_color_from_number(num: int) -> str:
    if num == 0: return 'RED_VIOLET'
    if num == 5: return 'GREEN_VIOLET'
    if num in [2, 4, 6, 8]: return 'RED'
    if num in [1, 3, 7, 9]: return 'GREEN'
    return 'RED' 

def get_color_emoji(color: str) -> str:
    if color == 'RED': return "🔴"
    if color == 'GREEN': return "🟢"
    if color == 'RED_VIOLET': return "🔴🟣"
    if color == 'GREEN_VIOLET': return "🟢🟣"
    return "⚪"

class FeatureEngineer:
    def __init__(self, window_size: int = 6):
        self.window = window_size
        self.scaler = StandardScaler()

    def extract_features(self, colors: List[str], numbers: List[int]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        if len(colors) < self.window * 4: return None, None, None
            
        X, y = [], []
        for i in range(len(colors) - self.window):
            row = []
            for j in range(self.window):
                row.extend([float(COLOR_MAP.get(colors[i+j], 0)), float(numbers[i+j])])
            X.append(row)
            y.append(COLOR_MAP.get(colors[i+self.window], 0))
            
        curr_feats = []
        for j in range(1, self.window + 1):
            curr_feats.extend([float(COLOR_MAP.get(colors[-j], 0)), float(numbers[-j])])
            
        return self.scaler.fit_transform(X), np.array(y), self.scaler.transform([curr_feats])

# =========================================================================
# 🧠 MODULE 4: THE 10 AI CORES (4 CLASSES)
# =========================================================================

def ensure_probs(probs_dict: Dict[int, float]) -> Dict[str, float]:
    return {
        'RED': float(probs_dict.get(0, 0.0)),
        'GREEN': float(probs_dict.get(1, 0.0)),
        'RED_VIOLET': float(probs_dict.get(2, 0.0)),
        'GREEN_VIOLET': float(probs_dict.get(3, 0.0))
    }

def get_base_probs() -> Dict[str, float]:
    return {'RED': 0.4, 'GREEN': 0.4, 'RED_VIOLET': 0.1, 'GREEN_VIOLET': 0.1}

class TreeEngines:
    def __init__(self):
        self.rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
        self.gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
        
    def predict(self, X: np.ndarray, y: np.ndarray, curr_X: np.ndarray) -> Tuple[Dict[str, float], Dict[str, float]]:
        try:
            self.rf.fit(X, y); self.gb.fit(X, y)
            rf_res = {cls: prob for cls, prob in zip(self.rf.classes_, self.rf.predict_proba(curr_X)[0])}
            gb_res = {cls: prob for cls, prob in zip(self.gb.classes_, self.gb.predict_proba(curr_X)[0])}
            return ensure_probs(rf_res), ensure_probs(gb_res)
        except: return get_base_probs(), get_base_probs()

class MarkovEngine:
    @staticmethod
    def predict(colors: List[str]) -> Dict[str, float]:
        if len(colors) < 10: return get_base_probs()
        trans = {c: {s: 0 for s in STATES} for c in STATES}
        for i in range(len(colors)-1): trans[colors[i]][colors[i+1]] += 1
        curr = colors[-1]
        tot = sum(trans[curr].values())
        if tot == 0: return get_base_probs()
        return {s: float(trans[curr][s]/tot) for s in STATES}

class CountBasedEngine: # For NGram & MonteCarlo
    @staticmethod
    def calc_probs(matches: List[str]) -> Dict[str, float]:
        if not matches: return get_base_probs()
        tot = len(matches)
        return {s: float(matches.count(s)/tot) for s in STATES}

class NGramEngine:
    @staticmethod
    def predict(colors: List[str], n: int = 4) -> Dict[str, float]:
        if len(colors) < n+1: return get_base_probs()
        pat = tuple(colors[-n:])
        matches = [colors[i+n] for i in range(len(colors)-n) if tuple(colors[i:i+n]) == pat]
        return CountBasedEngine.calc_probs(matches)

class MonteCarloEngine:
    @staticmethod
    def predict(colors: List[str]) -> Dict[str, float]:
        return CountBasedEngine.calc_probs(colors)

class BayesianEngine:
    def __init__(self): self.nb = GaussianNB()
    def predict(self, X: np.ndarray, y: np.ndarray, curr_X: np.ndarray) -> Dict[str, float]:
        try:
            self.nb.fit(X, y)
            res = {cls: prob for cls, prob in zip(self.nb.classes_, self.nb.predict_proba(curr_X)[0])}
            return ensure_probs(res)
        except: return get_base_probs()

class SimpleLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=16, batch_first=True)
        # 💡 [NEW] Output Layer 4 ခုသို့ ပြောင်းထားသည်
        self.fc = nn.Linear(16, 4) 
        
    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

class LSTMEngine:
    def predict(self, colors: List[str]) -> Dict[str, float]:
        if len(colors) < 50: return get_base_probs()
        try:
            data = [float(COLOR_MAP[c]) for c in colors[-50:]]
            X_t = torch.tensor(data[:-1], dtype=torch.float32).view(1, -1, 1)
            y_t = torch.tensor([data[-1]], dtype=torch.long)
            
            model = SimpleLSTM()
            optimizer = optim.Adam(model.parameters(), lr=0.01)
            criterion = nn.CrossEntropyLoss()
            
            model.train()
            for _ in range(5):
                optimizer.zero_grad()
                loss = criterion(model(X_t), y_t)
                loss.backward()
                optimizer.step()
            
            model.eval()
            curr_X = torch.tensor(data[1:], dtype=torch.float32).view(1, -1, 1)
            with torch.no_grad(): 
                logits = model(curr_X)
                probs = torch.softmax(logits, dim=1).squeeze().tolist()
            return {'RED': float(probs[0]), 'GREEN': float(probs[1]), 'RED_VIOLET': float(probs[2]), 'GREEN_VIOLET': float(probs[3])}
        except Exception: 
            return get_base_probs()

class MetaOptimizer:
    def __init__(self):
        self.weights = {'rf': 0.15, 'gb': 0.15, 'markov': 0.15, 'ngram': 0.15, 'monte': 0.10, 'bayes': 0.10, 'lstm': 0.20}

    def update(self, actual_color: str, past_preds: Dict[str, Dict[str, float]]) -> None:
        if not past_preds: return
        try:
            total_w = 0.0
            for model, probs in past_preds.items():
                predicted_prob = probs.get(actual_color, 0.0)
                error = 1.0 - predicted_prob
                if error < 0.6: self.weights[model] += 0.05
                else: self.weights[model] = max(0.01, self.weights[model] - 0.02)
                total_w += self.weights[model]
                
            if total_w > 0:
                for k in self.weights: self.weights[k] = float(self.weights[k] / total_w)
        except Exception: pass

# =========================================================================
# ⚙️ MODULE 5: MASTER ORCHESTRATOR
# =========================================================================
class UltraMasterEngine:
    def __init__(self):
        self.fe = FeatureEngineer()
        self.opt = MetaOptimizer()
        self.trees = TreeEngines()
        self.bayes = BayesianEngine()
        self.lstm = LSTMEngine()
        self.last_probs: Dict[str, Dict[str, float]] = {}

    def analyze(self, docs: List[Dict[str, Any]]) -> Tuple[str, float, Dict[str, Dict[str, float]]]:
        if len(docs) < 50: return "RED", 50.0, {}
        try:
            colors = [d.get('color', 'RED') for d in reversed(docs)]
            nums = [int(d.get('number', 0)) for d in reversed(docs)]
            
            X, y, curr_X = self.fe.extract_features(colors, nums)
            
            probs = {}
            probs['markov'] = MarkovEngine.predict(colors)
            probs['ngram'] = NGramEngine.predict(colors)
            probs['monte'] = MonteCarloEngine.predict(colors)
            probs['lstm'] = self.lstm.predict(colors)
            
            if X is not None and len(X) > 10:
                probs['rf'], probs['gb'] = self.trees.predict(X, y, curr_X)
                probs['bayes'] = self.bayes.predict(X, y, curr_X)
            else:
                probs['rf'] = probs['gb'] = probs['bayes'] = get_base_probs()
                
            self.last_probs = probs
            
            final_probs = {s: 0.0 for s in STATES}
            w = self.opt.weights
            for model_name, p_dict in probs.items():
                weight = w.get(model_name, 0.1)
                for c in final_probs: final_probs[c] += p_dict.get(c, 0.0) * weight
                
            final_pred = max(final_probs, key=final_probs.get)
            conf = min(max(float(final_probs[final_pred]) * 100, 25.0), 99.0) 
            
            return final_pred, round(conf, 1), self.last_probs
            
        except Exception as e:
            logger.error(f"Master Engine Error: {e}")
            return "RED", 50.0, {}

# =========================================================================
# 💰 MODULE 6: TELEGRAM UI & PRESENTATION
# =========================================================================
def get_color_emoji(color: str) -> str:
    if color == 'RED': return "🔴"
    if color == 'GREEN': return "🟢"
    if color == 'RED_VIOLET': return "🔴🟣"
    if color == 'GREEN_VIOLET': return "🟢🟣"
    return "⚪"

class UIManager:
    def __init__(self, bot_client: Bot):
        self.bot = bot_client

    async def broadcast_prediction(self, issue: str, pred: str, step: int, conf: float, top_engine: str) -> None:
        emoji = get_color_emoji(pred)
        msg = (
            f"<b>[COLOR-AI 10-CORE PRO]</b>\n"
            f"⏰ Period: <code>{issue}</code>\n"
            f"🎯 Prediction: {emoji} <b>{pred}</b> {step}x\n"
            f"📊 Confidence: {conf}%\n"
            f"🧠 Top Engine: <code>{top_engine.upper()}</code>"
        )
        try: 
            await self.bot.send_message(chat_id=Config.CHANNEL_ID, text=msg)
        except Exception as e: 
            logger.error(f"Prediction Send Error: {e}")

    async def broadcast_result(self, issue: str, pred: str, step: int, is_win: bool, actual_color: str, actual_num: int) -> None:
        win_str = "WIN ✅" if is_win else "LOSE ❌"
        res_emoji = get_color_emoji(actual_color)
        
        msg = (
            f"<b>🏆 SIX-LOTTERY RESULTS</b>\n\n"
            f"⏰ Period: <code>{issue}</code>\n"
            f"🎯 Choice: {get_color_emoji(pred)} {pred} {step}x\n"
            f"📊 Result: {res_emoji} <b>{win_str}</b> | ({actual_num})"
        )
        
        try: 
            # 1. Result စာသားကို အရင်ပို့မည်
            await self.bot.send_message(chat_id=Config.CHANNEL_ID, text=msg)
            
            # 2. 💡 [FIXED] အနိုင်/အရှုံး Sticker ပို့မည့် စနစ် ပြန်ထည့်ထားပါသည်
            if is_win and Config.WIN_STICKER:
                await self.bot.send_sticker(chat_id=Config.CHANNEL_ID, sticker=Config.WIN_STICKER)
            elif not is_win and Config.LOSE_STICKER:
                await self.bot.send_sticker(chat_id=Config.CHANNEL_ID, sticker=Config.LOSE_STICKER)
                
        except Exception as e: 
            logger.error(f"Result/Sticker Send Error: {e}")

# =========================================================================
# 🚀 MODULE 7: THE MAIN CONTROLLER LOOP
# =========================================================================
class ApplicationController:
    def __init__(self):
        self.db = DatabaseManager(Config.MONGO_URI)
        self.ai = UltraMasterEngine()
        self.ui = UIManager(bot)
        self.last_issue: Optional[str] = None
        self.lose_streak: int = 0

    async def fetch_api_data(self, session: aiohttp.ClientSession) -> Optional[Dict[str, Any]]:
        json_data = {
            'pageSize': 10, 'pageNo': 1, 'typeId': 30, 'language': 7, 
            'random': '029ef9cb888540c4b19e0c4b5eb49cbf', 
            'signature': 'C526273C4CCB81BBD1A1227C3DC5D025', 
            'timestamp': int(time.time())
        }
        for attempt in range(3):
            try:
                async with session.post(Config.API_URL, headers=Config.get_headers(), json=json_data, timeout=5.0) as r:
                    if r.status == 200: return await r.json()
            except: await asyncio.sleep(0.5)
        return None

    # 💡 [NEW] Smart Win Check Logic (အရောင်တူရင်ဖြစ်ဖြစ်၊ Half-Win ဖြစ်ဖြစ် နိုင်တယ်လို့ မှတ်ယူမည်)
    def check_win(self, predicted: str, actual: str) -> bool:
        if predicted == actual: return True
        if predicted == 'RED' and actual == 'RED_VIOLET': return True
        if predicted == 'GREEN' and actual == 'GREEN_VIOLET': return True
        return False

    async def run_forever(self) -> None:
        await self.db.initialize()
        
        async with aiohttp.ClientSession() as session:
            logger.info("🔥 COLOR-AI V2 GAME LOOP STARTED 🔥")
            while True:
                try:
                    data = await self.fetch_api_data(session)
                    if not data or data.get('code') != 0:
                        await asyncio.sleep(1.5); continue
                        
                    records = data.get("data", {}).get("list", [])
                    if not records: continue
                    
                    latest = records[0]
                    issue = str(latest["issueNumber"])
                    number = int(latest["number"])
                    color = get_color_from_number(number)
                    
                    if not self.last_issue:
                        self.last_issue = issue
                        recent_preds = await self.db.get_recent_predictions(15)
                        
                        self.lose_streak = 0
                        for p in recent_preds:
                            if p.get("win_lose") == "LOSE": self.lose_streak += 1
                            else: break
                        if self.lose_streak >= len(Config.MULTIPLIERS): self.lose_streak = 0

                        next_issue = str(int(issue) + 1)
                        docs = await self.db.get_history(500)
                        
                        pred, conf, details = self.ai.analyze(docs)
                        top_model = max(self.ai.opt.weights, key=self.ai.opt.weights.get) if self.ai.opt.weights else "rf"
                        
                        await self.ui.broadcast_prediction(next_issue, pred, self.lose_streak + 1, conf, top_model)
                        await asyncio.sleep(1.0); continue

                    if int(issue) > int(self.last_issue):
                        await self.db.save_history(issue, number, color)
                        self.ai.opt.update(color, self.ai.last_probs)
                        
                        pred_doc = await self.db.predictions.find_one({"issue_number": issue})
                        if pred_doc and pred_doc.get('predicted_color'):
                            predicted_color = str(pred_doc['predicted_color'])
                            
                            # 💡 Apply Smart Win Check Logic here
                            is_win = self.check_win(predicted_color, color)
                            win_lose_db = "WIN" if is_win else "LOSE"
                            
                            await self.db.update_result(issue, color, number, win_lose_db)
                            
                            current_step = self.lose_streak + 1
                            await self.ui.broadcast_result(issue, predicted_color, current_step, is_win, color, number)
                            
                            if is_win: self.lose_streak = 0
                            else: 
                                self.lose_streak += 1
                                if self.lose_streak >= len(Config.MULTIPLIERS): self.lose_streak = 0

                        self.last_issue = issue
                        
                        next_issue = str(int(issue) + 1)
                        docs = await self.db.get_history(500)
                        
                        pred, conf, details = self.ai.analyze(docs)
                        await self.db.save_prediction(next_issue, pred, conf, details)
                        
                        top_model = max(self.ai.opt.weights, key=self.ai.opt.weights.get)
                        current_step = self.lose_streak + 1
                        
                        await self.ui.broadcast_prediction(next_issue, pred, current_step, conf, top_model)

                except Exception as e:
                    logger.error(f"Critical Loop Exception: {e}")
                
                await asyncio.sleep(1.5)

async def main() -> None:
    await bot.delete_webhook(drop_pending_updates=True)
    asyncio.create_task(ApplicationController().run_forever())
    await dp.start_polling(bot)

if __name__ == '__main__':
    try: asyncio.run(main())
    except KeyboardInterrupt: pass
