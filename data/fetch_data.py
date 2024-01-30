import yfinance as yf

def fetch_stock_data(ticker, start_date, end_date):
    '''  
        params: ticker, start_date, end_date
        all are strings 
    '''
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        return stock_data
    except: 
        print("error in fetching data")
    