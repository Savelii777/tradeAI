#!/bin/bash
# Скрипт для создания архива для развертывания live trading

echo "📦 Создание архива для развертывания live trading..."

# Создаем временную директорию
DEPLOY_DIR="tradeAI_deploy_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$DEPLOY_DIR"

echo "📂 Копирование файлов..."

# 1. Модели (обученные)
mkdir -p "$DEPLOY_DIR/models/v8_improved"
cp -r models/v8_improved/* "$DEPLOY_DIR/models/v8_improved/" 2>/dev/null || echo "⚠️  Модели не найдены"

# 2. Конфигурация
mkdir -p "$DEPLOY_DIR/config"
cp config/settings.yaml "$DEPLOY_DIR/config/" 2>/dev/null
cp config/trading_params.yaml "$DEPLOY_DIR/config/" 2>/dev/null
cp config/risk_management.yaml "$DEPLOY_DIR/config/" 2>/dev/null
cp config/pairs_20.json "$DEPLOY_DIR/config/" 2>/dev/null
cp config/secrets.yaml.example "$DEPLOY_DIR/config/" 2>/dev/null
echo "secrets.yaml.example скопирован - ЗАПОЛНИТЕ secrets.yaml на сервере!" > "$DEPLOY_DIR/config/README_SECRETS.txt"

# 3. Исходный код
cp -r src "$DEPLOY_DIR/"

# 4. Скрипты live trading
mkdir -p "$DEPLOY_DIR/scripts"
cp scripts/live_trading_v10_csv.py "$DEPLOY_DIR/scripts/" 2>/dev/null
cp scripts/live_trading_mexc_v8.py "$DEPLOY_DIR/scripts/" 2>/dev/null
cp scripts/check_mexc_limits.py "$DEPLOY_DIR/scripts/" 2>/dev/null
cp scripts/preflight_check.py "$DEPLOY_DIR/scripts/" 2>/dev/null

# 5. Зависимости
cp requirements.txt "$DEPLOY_DIR/"

# 6. Создаем пустые директории для логов и данных
mkdir -p "$DEPLOY_DIR/logs"
mkdir -p "$DEPLOY_DIR/data/candles"
mkdir -p "$DEPLOY_DIR/results"
touch "$DEPLOY_DIR/active_trades.json"
echo "[]" > "$DEPLOY_DIR/active_trades.json"

# 7. README для развертывания
cat > "$DEPLOY_DIR/README_DEPLOY.md" << 'DEPLOY_README'
# 🚀 TradeAI Live Trading - Инструкция по развертыванию

## 📋 Предварительные требования
- Python 3.9+
- pip
- VPS/сервер (рекомендуется: Германия/Нидерланды для MEXC)

## 🔧 Шаги развертывания

### 1. Распаковка архива
```bash
tar -xzf tradeAI_deploy_*.tar.gz
cd tradeAI_deploy_*/
```

### 2. Установка зависимостей
```bash
pip install -r requirements.txt
```

### 3. Настройка конфигурации
```bash
# Скопируйте пример и заполните своими API ключами
cp config/secrets.yaml.example config/secrets.yaml
nano config/secrets.yaml
```

**ВАЖНО! Заполните в secrets.yaml:**
- MEXC API ключ и секрет
- Telegram bot token и chat_id (опционально)

### 4. Проверка готовности
```bash
# Проверка подключения к бирже
python scripts/preflight_check.py

# Проверка лимитов MEXC
python scripts/check_mexc_limits.py
```

### 5. Запуск Live Trading

**Режим с логированием в CSV:**
```bash
# Основной скрипт (рекомендуется)
python scripts/live_trading_v10_csv.py --balance 61 --max-positions 1
```

**Запуск в фоне (screen/tmux):**
```bash
# Создать screen сессию
screen -S tradeai

# Запустить trading
python scripts/live_trading_v10_csv.py --balance 61 --max-positions 1

# Отключиться: Ctrl+A, затем D
# Подключиться обратно: screen -r tradeai
```

**Запуск с systemd (автозапуск):**
```bash
sudo nano /etc/systemd/system/tradeai.service
```

Содержимое файла:
```ini
[Unit]
Description=TradeAI Live Trading Bot
After=network.target

[Service]
Type=simple
User=YOUR_USER
WorkingDirectory=/path/to/tradeAI_deploy_*/
ExecStart=/usr/bin/python3 scripts/live_trading_v10_csv.py --balance 61 --max-positions 1
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable tradeai
sudo systemctl start tradeai
sudo systemctl status tradeai
```

## 📊 Мониторинг

### Логи
```bash
# Просмотр последних логов
tail -f logs/live_trading_*.log

# Результаты сделок
cat results/trades_*.csv
```

### Активные позиции
```bash
cat active_trades.json
```

## ⚙️ Параметры запуска

```bash
python scripts/live_trading_v10_csv.py \
  --balance 61 \          # Начальный баланс
  --max-positions 1 \     # Максимум одновременных позиций
  --min-confidence 0.65 \ # Минимальная уверенность модели
  --check-interval 300    # Интервал сканирования (сек)
```

## 🔒 Безопасность
- ✅ Используйте API ключи только с правами на торговлю (без вывода)
- ✅ Установите IP whitelist в настройках MEXC API
- ✅ Регулярно проверяйте логи
- ✅ Установите уведомления в Telegram

## 🛑 Остановка
```bash
# Если запущено в screen
screen -r tradeai
# Затем Ctrl+C

# Если запущено через systemd
sudo systemctl stop tradeai
```

## 📞 Поддержка
- Логи: `logs/`
- Результаты: `results/`
- Активные сделки: `active_trades.json`
DEPLOY_README

# 8. Создаем скрипт быстрого старта для сервера
cat > "$DEPLOY_DIR/start.sh" << 'START_SCRIPT'
#!/bin/bash
# Быстрый старт live trading

echo "🚀 Запуск TradeAI Live Trading..."

# Проверка виртуального окружения
if [ ! -d "venv" ]; then
    echo "📦 Создание виртуального окружения..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Проверка secrets.yaml
if [ ! -f "config/secrets.yaml" ]; then
    echo "❌ ОШИБКА: config/secrets.yaml не найден!"
    echo "Скопируйте config/secrets.yaml.example и заполните своими API ключами"
    exit 1
fi

# Запуск
python scripts/live_trading_v10_csv.py --balance 61 --max-positions 1 --min-confidence 0.65
START_SCRIPT

chmod +x "$DEPLOY_DIR/start.sh"

# 9. Создаем .gitignore для сервера
cat > "$DEPLOY_DIR/.gitignore" << 'GITIGNORE'
config/secrets.yaml
*.log
logs/
data/candles/
active_trades*.json
__pycache__/
*.pyc
venv/
.DS_Store
GITIGNORE

echo "✅ Файлы скопированы в $DEPLOY_DIR"

# Создаем архив
ARCHIVE_NAME="${DEPLOY_DIR}.tar.gz"
tar -czf "$ARCHIVE_NAME" "$DEPLOY_DIR"

echo "✅ Архив создан: $ARCHIVE_NAME"
echo ""
echo "📦 Размер архива:"
du -h "$ARCHIVE_NAME"
echo ""
echo "🚀 Для развертывания на сервере:"
echo "   1. Скопируйте архив: scp $ARCHIVE_NAME user@server:/path/"
echo "   2. Распакуйте: tar -xzf $ARCHIVE_NAME"
echo "   3. Следуйте инструкциям в README_DEPLOY.md"
echo ""
echo "🔑 НЕ ЗАБУДЬТЕ:"
echo "   - Заполнить config/secrets.yaml с API ключами"
echo "   - Проверить config/settings.yaml"
echo "   - Запустить preflight_check.py перед live trading"

# Удаляем временную директорию
rm -rf "$DEPLOY_DIR"
