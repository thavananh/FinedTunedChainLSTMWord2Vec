import telegram
import asyncio

async def send_file():
    bot = telegram.Bot(token='7615198086:AAFdw5saYSgzxqpwrivDPklWvv9GFN3FcGQ')
    chat_id = '-4788105046'
    file_path = 'path/to/your_file.txt'

    # Mở file ở chế độ đọc nhị phân
    with open(file_path, 'rb') as file:
        await bot.send_document(chat_id=chat_id, document=file)
    
    print("File đã được gửi!")

if __name__ == '__main__':
    asyncio.run(send_file())
