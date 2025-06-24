# File: src/telegram_scraper.py

import os
import pandas as pd
from telethon.sync import TelegramClient
from dotenv import load_dotenv
import asyncio

# --- Load Environment Variables ---
# This will look for a .env file in the root directory and load its variables
load_dotenv()

# --- Get Credentials from .env file ---
api_id = os.getenv("API_ID")
api_hash = os.getenv("API_HASH")

# --- Check if credentials are set ---
if not api_id or not api_hash:
    raise ValueError("API_ID and API_HASH must be set in the .env file.")

# --- Expanded List of Channels ---
# Cleaned and formatted list from your input
channel_usernames = [
    'ZemenExpress', 'nevacomputer', 'meneshayeofficial', 'ethio_brand_collection',
    'Leyueqa', 'sinayelj', 'Shewabrand', 'helloomarketethiopia', 'modernshoppingcenter',
    'qnashcom', 'Fashiontera', 'kuruwear', 'gebeyaadama', 'MerttEka', 'forfreemarket',
    'classybrands', 'marakibrand', 'aradabrand2', 'marakisat2', 'belaclassic', 'AwasMart'
]

# --- Connect to Telegram ---
# 'anon' is a session name, it will create anon.session file to store your session
client = TelegramClient('anon', api_id, api_hash)

async def fetch_messages():
    """
    Asynchronously fetches the latest 200 messages from the specified Telegram channels
    and saves them to a CSV file.
    """
    all_messages = []
    async with client:
        for channel_username in channel_usernames:
            print(f"Fetching messages from @{channel_username}...")
            try:
                # Use client.get_input_entity to handle potential username issues
                channel_entity = await client.get_input_entity(channel_username)
                
                async for message in client.iter_messages(channel_entity, limit=200):
                    # We only care about messages that have text content
                    if message.text:
                        all_messages.append({
                            'channel': channel_username,
                            'message_id': message.id,
                            'text': message.text,
                            'date': message.date,
                            'views': message.views
                        })
            except Exception as e:
                print(f"Could not fetch messages from @{channel_username}. Reason: {e}")
    
    if not all_messages:
        print("No messages were collected. Check channel names and permissions.")
        return

    # Create a DataFrame and save it to the 'data' directory
    df = pd.DataFrame(all_messages)
    
    # Ensure the 'data' directory exists
    if not os.path.exists('../data'):
        os.makedirs('../data')
        
    df.to_csv('../data/raw_scraped_data.csv', index=False, encoding='utf-8')
    print(f"\nScraping complete. Collected {len(df)} messages.")
    print("Data saved to data/raw_scraped_data.csv")

# --- Run the Main Async Function ---
if __name__ == "__main__":
    # This structure allows the script to be run directly
    try:
        asyncio.run(fetch_messages())
    except ValueError as e:
        print(f"Error: {e}")
