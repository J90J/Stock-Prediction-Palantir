import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def get_current_sentiment(ticker_symbol="PLTR"):
    """
    Fetches the latest news for the given ticker and calculates a sentiment score.
    Returns a dictionary with 'score' (-1 to 1), 'verdict', and 'headlines'.
    """
    print(f"Analyzing news sentiment for {ticker_symbol}...")
    
    try:
        ticker = yf.Ticker(ticker_symbol)
        news = ticker.news
        
        if not news:
            return {"score": 0, "verdict": "NEUTRAL (No News)", "headlines": []}

        analyzer = SentimentIntensityAnalyzer()
        total_score = 0
        count = 0
        headlines = []

        for item in news:
            # Handle variable yfinance structure (sometimes nested in 'content')
            title = item.get('title')
            if not title and 'content' in item:
                title = item['content'].get('title')
            
            if not title:
                continue
            
            # Create a list of headlines for display
            headlines.append(title)
            
            # Analyze sentiment
            vs = analyzer.polarity_scores(title)
            compound = vs['compound']
            total_score += compound
            count += 1

        if count == 0:
            return {"score": 0, "verdict": "NEUTRAL", "headlines": []}

        avg_score = total_score / count
        
        # Determine verdict
        if avg_score >= 0.05:
            verdict = "POSITIVE"
        elif avg_score <= -0.05:
            verdict = "NEGATIVE"
        else:
            verdict = "NEUTRAL"
            
        print(f"Analyzed {count} headlines. Average Score: {avg_score:.2f}")
        
        return {
            "score": avg_score, 
            "verdict": verdict,
            "headlines": headlines[:3] # Return top 3 for display
        }

    except Exception as e:
        print(f"Error fetching sentiment: {e}")
        return {"score": 0, "verdict": "ERROR", "headlines": []}

if __name__ == "__main__":
    result = get_current_sentiment("PLTR")
    print("\nResult:")
    print(result)
