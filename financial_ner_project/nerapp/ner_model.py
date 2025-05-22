import os
import pandas as pd
from .financial_ner_class import FinancialNER

def load_financial_ner():
    """Load the financial NER model"""
    try:
        # Try to load the data files
        stocks_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'stocks.tsv')
        exchanges_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'stock_exchanges.tsv')
        
        print(f"Loading stocks from: {stocks_path}")
        print(f"Loading exchanges from: {exchanges_path}")
        
        stocks_df = pd.read_csv(stocks_path, sep='\t')
        exchanges_df = pd.read_csv(exchanges_path, sep='\t')
        
        print(f"Loaded {len(stocks_df)} stocks and {len(exchanges_df)} exchanges")
        return FinancialNER(stocks_df, exchanges_df)
        
    except FileNotFoundError as e:
        print(f"Data files not found: {e}")
        print("Creating empty FinancialNER instance...")
        # Create with empty DataFrames as fallback
        stocks_df = pd.DataFrame()
        exchanges_df = pd.DataFrame()
        return FinancialNER(stocks_df, exchanges_df)
    except Exception as e:
        print(f"Error loading data: {e}")
        # Create with empty DataFrames as fallback
        stocks_df = pd.DataFrame()
        exchanges_df = pd.DataFrame()
        return FinancialNER(stocks_df, exchanges_df)

# Create a singleton instance to avoid reloading data multiple times
_financial_ner_instance = None

def get_financial_ner_instance():
    """Get the singleton FinancialNER instance"""
    global _financial_ner_instance
    if _financial_ner_instance is None:
        print("Creating new FinancialNER instance...")
        _financial_ner_instance = load_financial_ner()
    return _financial_ner_instance

# For backward compatibility, create the financial_ner variable
financial_ner = get_financial_ner_instance()

def format_entities(entities):
    """Format extracted entities for display"""
    output = []

    if entities['symbols']:
      output.append("STOCK SYMBOLS:")
      for item in entities['symbols']:
          if isinstance(item, dict): 
            output.append(f"  • {item.get('symbol', '')} - {item.get('company', 'Unknown')} ({item.get('industry', 'Unknown')}, Market Cap: {item.get('market_cap', 'N/A')})")
          else:  # It's just a string
            output.append(f"  • {item}")


    if entities['companies']:
        output.append("\n COMPANIES:")
        for company in entities['companies']:
            output.append(f"  • {company}")

    if entities['exchanges']:
        output.append("\n STOCK EXCHANGES:")
        for exchange in entities['exchanges']:
            if 'country' in exchange:
                output.append(f"  • {exchange['name']} ({exchange['code']}, {exchange['country']})")
            else:
                output.append(f"  • {exchange['name']}")

    if entities['industries']:
        output.append("\n INDUSTRIES:")
        for industry in entities['industries']:
            output.append(f"  • {industry}")

    if entities['market_caps']:
        output.append("\n MARKET CAPITALIZATIONS:")
        for cap in entities['market_caps']:
            output.append(f"  • {cap}")

    return '\n'.join(output)

# Define the process_text function
def process_text(text, stocks_df, exchanges_df):
    ner = FinancialNER(stocks_df, exchanges_df)
    entities = ner.extract_entities(text)
    return format_entities(entities)