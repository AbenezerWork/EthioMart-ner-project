# File: src/telegram_scraper.py

import os
import pandas as pd
from telethon.sync import TelegramClient
from dotenv import load_dotenv
import asyncio

# --- Import your custom preprocessor ---
from text_preprocessor import AmharicPreprocessor

# --- Load Environment Variables ---
load_dotenv()

api_id = os.getenv("API_ID")
api_hash = os.getenv("API_HASH")

if not api_id or not api_hash:
    raise ValueError("API_ID and API_HASH must be set in the .env file.")

# --- List of Channels ---
channel_usernames = [
    'ZemenExpress', 'nevacomputer', 'meneshayeofficial', 'ethio_brand_collection',
    'Leyueqa', 'sinayelj', 'Shewabrand', 'helloomarketethiopia', 'modernshoppingcenter',
    'qnashcom', 'Fashiontera', 'kuruwear', 'gebeyaadama', 'MerttEka', 'forfreemarket',
    'classybrands', 'marakibrand', 'aradabrand2', 'marakisat2', 'belaclassic', 'AwasMart'
]

# --- Initialize the Telegram Client ---
client = TelegramClient('anon', api_id, api_hash)

async def fetch_and_structure_messages():
    """
    Fetches messages, processes them into a structured format, flattens the structure,
    and saves the result as a CSV file.
    """
    structured_data_list = []
    
    # --- Initialize the preprocessor once ---
    preprocessor = AmharicPreprocessor()

    async with client:
        for channel_username in channel_usernames:
            print(f"Fetching messages from @{channel_username}...")
            try:
                channel_entity = await client.get_input_entity(channel_username)
                
                async for message in client.iter_messages(channel_entity, limit=200):
                    if message.text:
                        # --- 1. Preprocess the message content ---
                        original_text = message.text
                        normalized_text = preprocessor.normalize_text(original_text)
                        cleaned_text = preprocessor.clean_text(normalized_text, remove_punc=False)
                        tokens = preprocessor.tokenize(cleaned_text)

                        # --- 2. Structure the data into our unified format ---
                        message_entry = {
                            "metadata.channel": channel_username,
                            "metadata.message_id": message.id,
                            "metadata.date": message.date.isoformat(),
                            "metadata.views": message.views,
                            "content.original_text": original_text,
                            "content.cleaned_text": cleaned_text,
                            "content.tokens": tokens
                        }
                        
                        structured_data_list.append(message_entry)

            except Exception as e:
                print(f"Could not fetch messages from @{channel_username}. Reason: {e}")
    
    if not structured_data_list:
        print("No messages were collected. Check channel names and permissions.")
        return

    # --- 3. Convert to DataFrame and save as CSV ---
    print("\nFlattening data and preparing to save as CSV...")
    
    # Convert the list of dictionaries into a pandas DataFrame
    # Using a pre-flattened dictionary structure simplifies this step.
    df = pd.DataFrame(structured_data_list)
    
    output_filename = '../data/structured_messages.csv'
    if not os.path.exists('../data'):
        os.makedirs('../data')
        
    # Save the DataFrame to a CSV file
    df.to_csv(output_filename, index=False, encoding='utf-8')

    print(f"\nProcessing complete. Collected and structured {len(df)} messages.")
    print(f"Data saved to {output_filename}")


if __name__ == "__main__":
    # You might need to install pandas if you haven't already: pip install pandas
    try:
        asyncio.run(fetch_and_structure_messages())
    except ValueError as e:
        print(f"Error: {e}")