import requests

TELEGRAM_TOKEN = "8270168075:AAHkJ_bbJGgk4fV3r0_Gc8NQb07O_zUMBJc"
TELEGRAM_CHAT_ID = "677822370"

def send_test_message():
    msg = "🔔 <b>Test Message</b>\n\nIf you see this, your Telegram bot is configured correctly!"
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': msg,
        'parse_mode': 'HTML'
    }
    
    print(f"Sending test message to {TELEGRAM_CHAT_ID}...")
    try:
        response = requests.post(url, data=data)
        if response.status_code == 200:
            print("✅ Success! Check your Telegram.")
        else:
            print(f"❌ Failed. Status: {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    send_test_message()
