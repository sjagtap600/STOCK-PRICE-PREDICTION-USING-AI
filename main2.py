import os
import datetime
import discord
from discord.ext import commands
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import requests
import asyncio
from datetime import datetime, timedelta
import random
import yfinance as yf
import plotly.express as px
from discord import Intents
import matplotlib.pyplot as plt
import aiocron
import sub_bot

# List of top stock companies
top_stock_companies = [
    'AAPL', 'GOOGL', 'TSLA', 'MSFT', 'AMZN', 'FB', 'BRK-B', 'SPY', 'BABA',
    'JPM', 'WMT', 'V', 'T', 'UNH', 'PFE', 'INTC', 'VZ', 'ORCL'
]

df = None
df_not_none = False
count = 0
random_company = ''
nrows = 0

# Create an 'images' directory if not exists
if not os.path.exists("images"):
    os.mkdir("images")

# Load Discord bot token
my_secret = os.environ.get('DISCORD_TOKEN')
if my_secret is None:
    print("DISCORD_TOKEN environment variable is not set.")

# Initialize Discord bot
intents = Intents.default()
intents.typing = False
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

# Events and commands
@bot.event
async def on_ready():
    print(f'{bot.user.name} has connected to Discord!')

