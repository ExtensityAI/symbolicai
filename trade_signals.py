"""
Financial News Trading Signal Analysis - SymbolicAI Coding Challenge
Author: Sang Hyeon Lee
Date: 2025

This script analyzes financial news and generates trading signals using SymbolicAI contracts.
"""

from typing import List
from pydantic import Field
from symai import Expression
from symai.strategy import contract
from symai.models import LLMDataModel


# Data Models
class NewsInput(LLMDataModel):
    """Input model for financial news analysis"""

    timestamp: str = Field(description="ISO timestamp when news was available")
    body: str = Field(description="The news article text content")
    company_name: str = Field(description="Name of the company the news relates to")


class TradingSignal(LLMDataModel):
    """Output model for trading signal analysis"""

    trade_signal: int = Field(
        description="Trading signal: 1 for buy, 0 for neutral, -1 for sell"
    )
    justification: str = Field(
        description="Detailed reasoning for the trading signal decision"
    )
    confidence: float = Field(description="Confidence level between 0.0 and 1.0")


# Main Contract Class
@contract(
    pre_remedy=True,  # Auto-fix input validation issues
    post_remedy=True,  # Auto-fix output validation issues
    verbose=True,  # Enable detailed logging for debugging
    accumulate_errors=True,  # Learn from previous correction attempts
)
class FinancialNewsAnalyzer(Expression):
    """
    Analyzes financial news and generates trading signals with validation.
    Uses SymbolicAI contracts to ensure reliable, structured outputs.
    """

    @property
    def prompt(self) -> str:
        return """You are an expert financial analyst specializing in news-driven trading signals.

Your task is to analyze financial news and generate trading recommendations based on:
1. Market sentiment analysis
2. Company-specific impact assessment  
3. Industry trends and implications
4. Risk factors and market conditions

Generate a trading signal:
- BUY (1): Strong positive news likely to drive stock price up
- NEUTRAL (0): Mixed or minimal impact on stock price
- SELL (-1): Negative news likely to drive stock price down

Provide clear justification and realistic confidence levels."""

    def pre(self, input: NewsInput) -> bool:
        """Validate input news data"""

        # Check timestamp format
        if not input.timestamp:
            raise ValueError("Timestamp cannot be empty")

        # Validate news body content
        if not input.body or len(input.body.strip()) < 20:
            raise ValueError(
                "News body must contain at least 20 characters of meaningful content"
            )

        # Check company name
        if not input.company_name or len(input.company_name.strip()) < 2:
            raise ValueError("Company name must be at least 2 characters long")

        # Check for spam or irrelevant content
        spam_indicators = ["click here", "buy now", "limited time", "free trial"]
        if any(spam in input.body.lower() for spam in spam_indicators):
            raise ValueError(
                "Input appears to contain promotional/spam content rather than legitimate financial news"
            )

        return True

    def act(self, input: NewsInput, **kwargs) -> NewsInput:
        """
        Preprocess the news data for better analysis
        This step could include text cleaning, entity extraction, etc.
        """

        # Clean and normalize the text
        cleaned_body = input.body.strip()

        # Remove excessive whitespace
        cleaned_body = " ".join(cleaned_body.split())

        # Normalize company name
        normalized_company = input.company_name.strip().title()

        # Return preprocessed input
        return NewsInput(
            timestamp=input.timestamp,
            body=cleaned_body,
            company_name=normalized_company,
        )

    def post(self, output: TradingSignal) -> bool:
        """Validate the generated trading signal"""

        # Validate trade signal values
        if output.trade_signal not in [-1, 0, 1]:
            raise ValueError(
                "Trade signal must be exactly -1 (sell), 0 (neutral), or 1 (buy)"
            )

        # Validate justification quality
        if not output.justification or len(output.justification.strip()) < 30:
            raise ValueError(
                "Justification must be at least 30 characters and provide meaningful reasoning"
            )

        # Check for generic/template responses
        generic_phrases = [
            "based on the news",
            "this could affect",
            "market conditions",
            "further analysis needed",
            "depends on market",
        ]
        if all(
            phrase in output.justification.lower() for phrase in generic_phrases[:2]
        ):
            raise ValueError(
                "Justification appears too generic. Provide specific analysis of the news content."
            )

        # Validate confidence range
        if not (0.0 <= output.confidence <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")

        # Logical consistency checks
        if abs(output.trade_signal) == 1 and output.confidence < 0.3:
            raise ValueError("Strong buy/sell signals should have confidence >= 0.3")

        if output.trade_signal == 0 and output.confidence > 0.8:
            raise ValueError(
                "Neutral signals with very high confidence (>0.8) seem inconsistent"
            )

        return True

    def forward(self, input: NewsInput, **kwargs) -> TradingSignal:
        """
        Main execution logic - always runs regardless of contract success
        """

        if self.contract_successful:
            # Contract validation passed - return the validated result
            print(f"✅ Contract validation successful for {input.company_name}")
            return self.contract_result
        else:
            # Contract failed - provide fallback response
            print(
                f"⚠️ Contract validation failed for {input.company_name}. Using fallback logic."
            )
            print(f"Error: {getattr(self, 'contract_exception', 'Unknown error')}")

            # Implement simple fallback logic
            fallback_signal = 0  # Default to neutral
            fallback_confidence = 0.1  # Low confidence

            # Basic sentiment analysis as fallback
            positive_keywords = [
                "profit",
                "growth",
                "success",
                "increase",
                "positive",
                "record",
                "strong",
            ]
            negative_keywords = [
                "loss",
                "decline",
                "investigation",
                "fall",
                "negative",
                "weak",
                "layoffs",
            ]

            body_lower = input.body.lower()
            positive_count = sum(1 for word in positive_keywords if word in body_lower)
            negative_count = sum(1 for word in negative_keywords if word in body_lower)

            if positive_count > negative_count:
                fallback_signal = 1
                fallback_justification = f"Fallback analysis detected {positive_count} positive indicators in the news."
            elif negative_count > positive_count:
                fallback_signal = -1
                fallback_justification = f"Fallback analysis detected {negative_count} negative indicators in the news."
            else:
                fallback_justification = (
                    "Fallback analysis could not determine clear sentiment direction."
                )

            return TradingSignal(
                trade_signal=fallback_signal,
                justification=fallback_justification,
                confidence=fallback_confidence,
            )


def process_news_batch(news_list: List[dict]) -> List[dict]:
    """
    Process a batch of news articles and return trading signals

    Args:
        news_list: List of news dictionaries with timestamp, body, company_name

    Returns:
        List of dictionaries with trade_signal, justification, confidence
    """

    analyzer = FinancialNewsAnalyzer()
    results = []

    print(f"\n🔍 Processing {len(news_list)} news articles...\n")

    for i, news_data in enumerate(news_list, 1):
        try:
            print(f"[{i}/{len(news_list)}] Analyzing: {news_data['company_name']}")

            # Create input model
            news_input = NewsInput(
                timestamp=news_data["timestamp"],
                body=news_data["body"],
                company_name=news_data["company_name"],
            )

            # Process with contract
            result = analyzer(input=news_input)

            # Convert to output dictionary
            output_dict = {
                "trade_signal": result.trade_signal,
                "justification": result.justification,
                "confidence": result.confidence,
            }

            results.append(output_dict)

            # Print result summary
            signal_text = {-1: "SELL", 0: "NEUTRAL", 1: "BUY"}[result.trade_signal]
            print(f"   Result: {signal_text} (confidence: {result.confidence:.2f})")
            print(f"   Reason: {result.justification[:80]}...")
            print()

        except Exception as e:
            print(f"   ❌ Error processing {news_data['company_name']}: {str(e)}")
            # Add error result
            results.append(
                {
                    "trade_signal": 0,
                    "justification": f"Processing error: {str(e)}",
                    "confidence": 0.0,
                }
            )
            print()

    return results


def main():
    example_inputs = [
        {
            "timestamp": "2025-08-07T09:30:00Z",
            "body": "Zentronix Corp announced record-breaking quarterly profits, surpassing analyst expectations by 30%. The CEO attributes the growth to strong overseas demand and successful product launches.",
            "company_name": "Zentronix Corp",
        },
        {
            "timestamp": "2025-08-07T13:45:00Z",
            "body": "Novatek Industries is currently under investigation for potential accounting irregularities. Market reaction has been swift, with shares falling 12% in early trading.",
            "company_name": "Novatek Industries",
        },
        {
            "timestamp": "2025-08-07T15:00:00Z",
            "body": "GloboTech has announced a major restructuring initiative aimed at improving efficiency. While layoffs are expected, investors are cautiously optimistic about long-term gains.",
            "company_name": "GloboTech",
        },
    ]
    print("🚀 Financial News Trading Signal Analysis")
    print("=" * 50)

    # Process the example inputs
    results = process_news_batch(example_inputs)

    print("\n📊 FINAL RESULTS SUMMARY")
    print("=" * 50)

    for i, (input_data, result) in enumerate(zip(example_inputs, results), 1):
        signal_text = {-1: "SELL", 0: "NEUTRAL", 1: "BUY"}.get(
            result["trade_signal"], "UNKNOWN"
        )
        print(f"\n{i}. {input_data['company_name']}")
        print(f"   Signal: {signal_text} ({result['trade_signal']})")
        print(f"   Confidence: {result['confidence']:.2f}")
        print(f"   Justification: {result['justification']}")

    print(f"\n✅ Processing complete! Analyzed {len(results)} articles.")

    return results


if __name__ == "__main__":
    main()
