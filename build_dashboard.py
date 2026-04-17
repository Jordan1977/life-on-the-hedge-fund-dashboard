# Complete Python Analytics Engine and HTML Generation Code

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sample DataFrame

# Generate sample data

# Replace with actual data source

stocks_data = {
    'Stock': ['AAPL', 'GOOGL', 'MSFT'],
    'Price': [150, 2800, 300],
    'Volume': [100000, 150000, 200000]
}

df = pd.DataFrame(stocks_data)

# Function to generate analytics

def generate_analytics(df):
    desc = df.describe()
    print("Analytics Summary:\n", desc)
    return desc

# Function to generate plots

def generate_plots(df):
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Stock', y='Price')
    plt.title('Stock Prices')
    plt.savefig('stocks_prices.png')
    plt.show()

# Function to save HTML report

def save_html_report(analytics, filename='report.html'):
    with open(filename, 'w') as f:
        f.write('<html><head><title>Stock Report</title></head><body>')
        f.write('<h1>Stock Analysis Report</h1>')
        f.write('<h2>Analytics Summary</h2>')
        f.write('<pre>{}</pre>'.format(analytics))
        f.write('<img src="stocks_prices.png" alt="Stock Prices">')
        f.write('</body></html>')

# Usage
if __name__ == '__main__':
    analytics = generate_analytics(df)
    generate_plots(df)
    save_html_report(analytics)
