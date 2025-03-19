echo "# CryptoAI" | Out-File -FilePath README.md -Encoding utf8
echo "" | Add-Content -Path README.md
echo "A trading bot for Bitcoin using the Binance API. This project fetches real-time Bitcoin prices, makes trading predictions using machine learning models, and executes trades based on predefined strategies." | Add-Content -Path README.md
echo "" | Add-Content -Path README.md
echo "## Setup" | Add-Content -Path README.md
echo "" | Add-Content -Path README.md
echo "### Prerequisites" | Add-Content -Path README.md
echo "- Python 3.8+" | Add-Content -Path README.md
echo "- Git" | Add-Content -Path README.md
echo "- Binance account with API keys" | Add-Content -Path README.md
echo "" | Add-Content -Path README.md
echo "### Installation" | Add-Content -Path README.md
echo "1. Clone the repository:" | Add-Content -Path README.md
echo "   ```bash" | Add-Content -Path README.md
echo "   git clone https://github.com/CrazyNeckline/CryptoAI.git" | Add-Content -Path README.md
echo "   cd CryptoAI" | Add-Content -Path README.md
echo "   ```" | Add-Content -Path README.md
echo "2. Install dependencies:" | Add-Content -Path README.md
echo "   ```bash" | Add-Content -Path README.md
echo "   pip install -r requirements.txt" | Add-Content -Path README.md
echo "   ```" | Add-Content -Path README.md
echo "3. Create a `.env` file with your Binance API keys:" | Add-Content -Path README.md
echo "   ```bash" | Add-Content -Path README.md
echo "   echo \"BINANCE_API_KEY=your_api_key\" >> .env" | Add-Content -Path README.md
echo "   echo \"BINANCE_API_SECRET=your_api_secret\" >> .env" | Add-Content -Path README.md
echo "   ```" | Add-Content -Path README.md
echo "4. Run the application:" | Add-Content -Path README.md
echo "   ```bash" | Add-Content -Path README.md
echo "   python app.py" | Add-Content -Path README.md
echo "   ```" | Add-Content -Path README.md
echo "5. Open the dashboard in your browser at `http://127.0.0.1:5000`." | Add-Content -Path README.md
echo "" | Add-Content -Path README.md
echo "## Usage" | Add-Content -Path README.md
echo "- The dashboard (`/`) displays the current Bitcoin price, trading stats, and recent trades." | Add-Content -Path README.md
echo "- The history page (`/history`) shows all past trades." | Add-Content -Path README.md
echo "- The bot automatically trades based on predictions from RandomForest models." | Add-Content -Path README.md
echo "" | Add-Content -Path README.md
echo "## License" | Add-Content -Path README.md
echo "This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details." | Add-Content -Path README.md