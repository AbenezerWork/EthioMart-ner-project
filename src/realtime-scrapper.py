# File: src/realtime_scraper.py

import os
import csv
import asyncio
from datetime import datetime
from telethon import TelegramClient, events
from dotenv import load_dotenv

# --- Load Environment Variables ---
load_dotenv()

api_id = os.getenv("API_ID")
api_hash = os.getenv("API_HASH")

if not api_id or not api_hash:
    raise ValueError("API_ID and API_HASH must be set in the .env file.")

# --- Define the output file for real-time data ---
OUTPUT_FILE = '../data/realtime_scraped_data.csv'
FILE_EXISTS = os.path.exists(OUTPUT_FILE)

# --- Expanded List of Channels ---
# Usernames of the channels to listen to
channel_usernames = [
    'ZemenExpress', 'nevacomputer', 'meneshayeofficial', 'ethio_brand_collection',
    'Leyueqa', 'sinayelj', 'Shewabrand', 'helloomarketethiopia', 'modernshoppingcenter',
    'qnashcom', 'Fashiontera', 'kuruwear', 'gebeyaadama', 'MerttEka', 'forfreemarket',
    'classybrands', 'marakibrand', 'aradabrand2', 'marakisat2', 'belaclassic', 'AwasMart'
]

# --- Initialize the Telegram Client ---
# The session name 'realtime_listener' will create a new .session file
client = TelegramClient('realtime_listener', api_id, api_hash)

def write_to_csv(data):
    """Appends a single row of data to the CSV file."""
    with open(OUTPUT_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(data)

async def main():
    """The main function that connects the client and runs indefinitely."""
    print("Real-time scraper started. Listening for new messages...")
    print(f"Monitoring {len(channel_usernames)} channels. Press Ctrl+C to stop.")

    # Write the header row only if the file is newly created
    if not FILE_EXISTS:
        write_to_csv(['channel', 'message_id', 'text', 'date', 'views'])

    # Connect the client
    await client.start()
    
    # Keep the script running until it is manually stopped (e.g., with Ctrl+C)
    await client.run_until_disconnected()

# --- This is the Event Handler ---
@client.on(events.NewMessage(chats=channel_usernames))
async def handle_new_message(event):
    """
    This function is triggered automatically by Telethon whenever a
    new message is sent to any of the channels in `channel_usernames`.
    """
    message = event.message
    
    # Get the username of the channel where the message was posted
    channel_info = await client.get_entity(message.peer_id)
    channel_username = channel_info.username

    # Format the current time
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print("-" * 30)
    print(f"[{timestamp}] New Message from @{channel_username}")
    
    # Prepare data for CSV
    data_row = [
        channel_username,
        message.id,
        message.text.replace('\n', ' '), # Replace newlines to keep CSV clean
        message.date,
        message.views
    ]
    
    # Append the new message data to our CSV file
    write_to_csv(data_row)
    print(f"Message saved to {OUTPUT_FILE}")
    print("-" * 30)


# --- Run the main function ---
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Scraper stopped manually.")
    except Exception as e:
        print(f"An error occurred: {e}")