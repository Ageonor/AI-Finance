# AI Finance: AI‑Powered Personal Finance Advisor

This repository contains a Streamlit‑based personal finance management application that leverages
AI/ML techniques to help users track spending, set goals, analyse trends and get investment
insights.  The project is organised into several pages that work together to deliver a
comprehensive financial dashboard.

## Features

### 🚀 AI‑Powered Interface

* A beautiful landing page (`Homepage.py`) built with Streamlit that introduces the app and
  highlights key features.  It uses animated cards and quick‑start buttons to guide users to
  different parts of the application.

### 📝 Data Input & Management

* Manual transaction entry with category selection and validation.
* CSV upload with a sample template and checks for required columns (`date`, `description`,
  `amount`, `category`, `type`).
* Automatic summary of total income, expenses and balance.
* Ability to download or clear entered records.

### 📊 Financial Analysis & Insights

* Combined view of manually entered data and uploaded CSV data.
* Calculation of key metrics: total income, total expenses, net balance and savings rate.
* Smart spending alerts (e.g. overspending, top expense categories, large transactions).
* Interactive charts comparing monthly income vs. expenses and spending by category.
* Visualisations built with Plotly.

### 🎯 Goal Management

* Create financial goals with target amount, date, category and priority.
* Intelligent suggestions for monthly savings based on your target date and amount.
* Track progress with progress bars, time remaining and required daily savings.
* Manage multiple goals – update or delete them as needed.

### 💎 Investment Insights

* Fetch live stock, index and commodity data using
  [yfinance](https://github.com/ranaroussi/yfinance).
* Display current price, daily change and volume for popular Indian equities (Nifty 50,
  Sensex, Reliance, TCS, HDFC Bank and more) as well as select global commodities (gold,
  silver, crude oil, bitcoin and ethereum).
* Compute technical indicators: simple moving averages (SMA), relative strength index
  (RSI), moving average convergence divergence (MACD) and Bollinger Bands.
* Perform a simple price prediction using linear regression with scikit‑learn.
* Maintain a watchlist and a portfolio in the Streamlit session state.

### 🤖 AI Financial Advisor Chat *(proof of concept)*

* The codebase includes a prototype chat interface demonstrating how to integrate
  personalised financial context into a large‑language‑model (LLM) prompt.  The
  `AI Financial Advisor Chat` page is commented out by default and requires a valid API key
  to use the Gemini API.  Uncomment and adapt this code at your own discretion.

## Project Structure

```
AI Finance/
├── Homepage.py                # Landing page for the app
├── finance.db                 # SQLite database (sample data)
├── Pages/
│   ├── Data Input & Management.py
│   ├── Financial Analysis & Insights.py
│   ├── Goal Management.py
│   ├── Investment Insights.py
│   └── AI Financial Advisor Chat.py
└── …
```

* `Homepage.py` defines the front page with animated cards, feature descriptions and quick
  start buttons.
* Each file inside `Pages/` corresponds to a tab in the Streamlit app and encapsulates a
  feature.
* `finance.db` is a simple SQLite database included for demonstration; you can delete it or
  replace it with your own data store.

## Requirements

Install the required Python packages listed below (minimum versions):

```
streamlit>=1.0
pandas>=1.4
numpy>=1.20
plotly>=5.0
yfinance
scikit-learn
requests
python-dateutil
googletrans==4.0.0-rc1  # optional, used in the chat prototype
```

These dependencies are captured in the [`requirements.txt`](requirements.txt) file.  You
can install them via pip:

```bash
pip install -r requirements.txt
```

## Running the App

1. **Clone this repository.**

   ```bash
   git clone https://github.com/Ageonor/AI-Finance.git
   cd AI-Finance
   ```

2. **Install dependencies.**

   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Streamlit app.**

   ```bash
   streamlit run "AI Finance/Homepage.py"
   ```

Visit the local URL provided by Streamlit (usually <http://localhost:8501>) to interact
with the application.

> **Note:** The investment page fetches live market data from Yahoo Finance.  Ensure you have
> an active internet connection.  If you are running behind a proxy or firewall, you may
> need to configure `yfinance` accordingly.  Some data sources may be unavailable in
> certain regions.

## Contributing

Contributions are welcome!  Feel free to open an issue or submit a pull request for bug
fixes, improvements or new features.

## License

This project is released under the MIT License.  See the [LICENSE](LICENSE) file for
details.