import pandas as pd
import nltk
import re
import string
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, ne_chunk
from nltk.tree import Tree
import numpy as np

# Downloading resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker_tab')


class FinancialNER:
    def __init__(self, stocks_df, exchanges_df):
        self.stocks_df = stocks_df
        self.exchanges_df = exchanges_df
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

        self.symbols = set(x for x in stocks_df['Symbol'].dropna().str.lower().tolist() if isinstance(x, str))

        # Clean suffixes from company names
        self.suffix_pattern = re.compile(r'\b(inc|inc\.|corporation|corp|co|ltd|llc|plc|group|technologies|technology|industries|industry|holdings|reit|limited|partners|sa|ag|nv)\b', re.IGNORECASE)

        # Clean industries
        industries_raw = stocks_df['Industry'].dropna().tolist()

        industries = set()

        for industry in industries_raw:
          if isinstance(industry, str):
            # Split on comma, ampersand (&), and slash (/)
            parts = re.split(r'[,&/]', industry)
            # Clean and lower each part, ignore empty strings
            parts = [part.strip().lower() for part in parts if part.strip()]
            industries.update(parts)

        self.industries = industries

        # Dictionary: {cleaned name: original name}
        self.cleaned_company_map = {}
        for name in stocks_df['CompanyName'].dropna().str.lower().unique():
            if isinstance(name, str):
                cleaned = self.suffix_pattern.sub('', name).strip()
                self.cleaned_company_map[cleaned] = name

        self.cleaned_company_names = set(self.cleaned_company_map.keys())

        # Exchange data processing
        self.exchange_codes = set(exchanges_df['BloombergExchangeCode'].str.lower().tolist())
        self.exchange_names = set()
        for desc in exchanges_df['Description'].dropna():
            self.exchange_names.add(desc.lower())
            parts = desc.split()
            if len(parts) > 2:
                self.exchange_names.add(' '.join(parts[:2]).lower())

        self.symbol_pattern = re.compile(r'\b[A-Z]{1,5}\b')
        self.ticker_pattern = re.compile(r'\$[A-Z]{1,5}\b')

    def preprocess_text(self, text):
        """Preprocess text for NER"""
        original_sentences = sent_tokenize(text)
        lowered_sentences = []
        processed_sentences = []

        for sentence in original_sentences:
            lowered = sentence.lower()
            lowered_sentences.append(lowered)

            tokens = word_tokenize(lowered)
            tokens = [token for token in tokens if token not in string.punctuation and not token.isdigit()]
            tokens = [token for token in tokens if token not in self.stop_words]
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]

            processed_sentences.append(tokens)

        return original_sentences, lowered_sentences, processed_sentences
    
    def _extract_symbols(self, sentence):
        """Extract stock symbols from text"""
        symbols = set()  # Use a set instead of a list to avoid duplicates

        # Only match symbols that are in our dataset
        # Check for ticker symbols with $ prefix
        ticker_matches = self.ticker_pattern.findall(sentence)
        for match in ticker_matches:
            symbol = match[1:].upper()  # Remove $ and convert to uppercase
            if symbol in self.stocks_df['Symbol'].values:
                symbols.add(symbol)

        # Check for symbols in parentheses (common format)
        parentheses_pattern = re.compile(r'\(([A-Z]{1,5})\)')
        parentheses_matches = parentheses_pattern.findall(sentence)
        for match in parentheses_matches:
            if match in self.stocks_df['Symbol'].values:
                symbols.add(match)

        # Only check for standalone symbols if they're in our dataset and at least 2 chars
        symbol_matches = self.symbol_pattern.findall(sentence)
        for match in symbol_matches:
            if len(match) >= 2 and match in self.stocks_df['Symbol'].values:
                symbols.add(match)

        return list(symbols)  # Convert back to list at the end

    def _extract_companies(self, sentence):
        """Extract company names using cleaned name and n-gram matching."""
        sentence_lower = sentence.lower()
        tokens = word_tokenize(sentence_lower)

        # Generate n-grams (up to trigrams)
        ngrams = []
        for n in range(1, 4):
            ngrams.extend([' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)])

        matched_companies = set()

        valid_rows = self.stocks_df[self.stocks_df['CompanyName'].notna()]

        for phrase in ngrams:
            if phrase in self.cleaned_company_names:
                original_name = self.cleaned_company_map[phrase]
                match = valid_rows[valid_rows['CompanyName'].str.lower() == original_name]

                if not match.empty:
                    matched_companies.add(match.iloc[0]['CompanyName'])

        return list(matched_companies)

    def _extract_exchanges(self, sentence):
        """Extract stock exchanges from text"""
        exchanges = []
        sentence_lower = sentence.lower()
        words = word_tokenize(sentence_lower)                                            
        word_set = set(words)
        
        # Check for context
        exchange_keywords = ['exchange', 'stock exchange', 'market']
        has_exchange_context = any(keyword in sentence_lower for keyword in exchange_keywords)
        
        # Match exchange codes
        for code in self.exchange_codes:
            # Skip non-string values
            if not isinstance(code, str):
                continue
            
            if code.lower() in word_set:
                # Use .notna() to filter out NaN values before applying string operations
                valid_rows = self.exchanges_df[self.exchanges_df['BloombergExchangeCode'].notna()]
                match = valid_rows[valid_rows['BloombergExchangeCode'].str.lower() == code.lower()]
                
                if not match.empty:
                    matched_code = match.iloc[0]['BloombergExchangeCode']
                    exchanges.append(matched_code)

        # Match exchange names if context exists
        if has_exchange_context:
            for name in self.exchange_names:
                # Skip non-string values
                if not isinstance(name, str):
                    continue
                    
                if name in sentence_lower:
                    # Use .notna() to filter out NaN values before applying string operations
                    valid_rows = self.exchanges_df[self.exchanges_df['Description'].notna()]
                    # Use safe string contains with na=False to avoid NaN issues
                    match = valid_rows[valid_rows['Description'].str.lower().str.contains(name, regex=False, na=False)]
                    
                    if not match.empty:
                        matched_name = match.iloc[0]['Description']
                        exchanges.append(matched_name)

        # Remove duplicates
        return list(dict.fromkeys(exchanges))
    
    def _extract_industries(self, sentence):
        industries = []
        print(f"sentence in industry: {sentence}")

        # Make sure self.industries is clean and contains only strings
        for industry in self.industries:
            # Skip non-string values or empty strings
            if not isinstance(industry, str) or not industry.strip():
                continue
            if "p" == industry:
                continue
                
            # Skip single character industries like "p" which likely cause false positives
            if len(industry.strip()) <= 1:
                continue
                
            # Now check if the industry appears in the sentence
            if industry in sentence:
                print(f"industry:{industry}")
                industries.append(industry.title())

        return industries


    def _extract_market_caps(self, sentence):
        market_caps = []

        # Pattern for market cap values (e.g., $10B, 5.2 billion, 800M) -> ('2.5', '.5', 'Trillion')
        market_cap_pattern = re.compile(r'\$?\s*(\d+(\.\d+)?)\s*(trillion|billion|million|t|b|m|bn|mln)\b', re.IGNORECASE)
        matches = market_cap_pattern.findall(sentence)

        for match in matches:
            value = float(match[0]) # converts 0 index to float
            unit = match[2].lower() # unit to lower case

            # Convert to standard form
            if unit in ('trillion', 't'):
                value *= 1_000_000_000_000
            elif unit in ('billion', 'b', 'bn'):
                value *= 1_000_000_000
            elif unit in ('million', 'm', 'mln'):
                value *= 1_000_000

            market_caps.append(f"${value:,.2f}")

        return market_caps
    

    def extract_entities(self, text):
        original_sentences, lowered_sentences, processed_sentences = self.preprocess_text(text)

        entities = {
            'companies': [],
            'symbols': [],
            'exchanges': [],
            'industries': [],
            'market_caps': []
        }

        # NLTK's built-in NER
        # nltk_entities = self._extract_nltk_entities(text)
        # if nltk_entities.get('organizations'):
        #     entities['companies'].extend(nltk_entities['organizations'])

        # Custom rule-based extraction
        for i in range(len(original_sentences)):
            original_sentence = original_sentences[i]
            lowered_sentence = lowered_sentences[i]
            processed_tokens = processed_sentences[i]

            # Use original case sentence for symbols
            symbols = self._extract_symbols(original_sentence)
            entities['symbols'].extend(symbols)

            # Use tokenized/lowered sentences for others
            companies = self._extract_companies(lowered_sentence)
            entities['companies'].extend(companies)

            exchanges = self._extract_exchanges(lowered_sentence)
            entities['exchanges'].extend(exchanges)

            industries = self._extract_industries(lowered_sentence)
            entities['industries'].extend(industries)

            market_caps = self._extract_market_caps(lowered_sentence)
            entities['market_caps'].extend(market_caps)

        # remove duplicates
        for entity_type in entities:
            entities[entity_type] = list(dict.fromkeys(entities[entity_type]))

        # Enrich entities with additional information
        # self._enrich_entities(entities)

        return entities
    

