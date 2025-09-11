import logging
from typing import List, Dict, Any, Optional
from enum import IntEnum
from dataclasses import dataclass
from pydantic import Field, validator
from symai import Expression
from symai.strategy import contract
from symai.models import LLMDataModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TradeSignalEnum(IntEnum):
    SELL = -1
    NEUTRAL = 0
    BUY = 1


@dataclass
class ValidationConfig:
    MIN_JUSTIFICATION_LENGTH: int = 30
    MIN_CONFIDENCE_FOR_STRONG_SIGNAL: float = 0.3
    MAX_CONFIDENCE_FOR_NEUTRAL: float = 0.8
    MIN_NEWS_BODY_LENGTH: int = 20
    MIN_COMPANY_NAME_LENGTH: int = 2


class NewsInput(LLMDataModel):
    timestamp: str = Field(
        description="ISO timestamp when news was available",
        pattern=r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z",
    )
    body: str = Field(description="The news article text content", min_length=20)
    company_name: str = Field(
        description="Name of the company the news relates to", min_length=2
    )

    @validator("body")
    def validate_body_content(cls, v):
        spam_indicators = ["click here", "buy now", "limited time", "free trial"]
        if any(spam in v.lower() for spam in spam_indicators):
            raise ValueError(
                "Content appears to be promotional/spam rather than financial news"
            )
        return v.strip()

    @validator("company_name")
    def normalize_company_name(cls, v):
        return v.strip().title()


class TradingSignal(LLMDataModel):
    trade_signal: int = Field(
        description="Trading signal: 1 for buy, 0 for neutral, -1 for sell", ge=-1, le=1
    )
    justification: str = Field(
        description="Detailed reasoning for the trading signal decision", min_length=30
    )
    confidence: float = Field(
        description="Confidence level between 0.0 and 1.0", ge=0.0, le=1.0
    )

    @validator("trade_signal")
    def validate_trade_signal_values(cls, v):
        if v not in [-1, 0, 1]:
            raise ValueError(
                "Trade signal must be exactly -1 (sell), 0 (neutral), or 1 (buy)"
            )
        return v

    @validator("justification")
    def validate_justification_quality(cls, v):
        if not v or len(v.strip()) < 30:
            raise ValueError("Justification must be at least 30 characters")

        # Check for overly generic responses
        generic_phrases = [
            "based on the news",
            "market conditions",
            "further analysis needed",
        ]
        if sum(phrase in v.lower() for phrase in generic_phrases) >= 2:
            raise ValueError("Justification is too generic - provide specific analysis")

        return v

    @validator("confidence")
    def validate_confidence_logic(cls, v, values):
        if "trade_signal" in values:
            signal = values["trade_signal"]
            if abs(signal) == 1 and v < 0.3:
                raise ValueError(
                    "Strong buy/sell signals should have confidence >= 0.3"
                )
            if signal == 0 and v > 0.8:
                raise ValueError(
                    "Neutral signals with very high confidence (>0.8) seem inconsistent"
                )
        return v


class FallbackAnalyzer:
    POSITIVE_KEYWORDS = [
        "profit",
        "growth",
        "success",
        "increase",
        "positive",
        "record",
        "strong",
        "beat",
        "exceed",
        "outperform",
    ]

    NEGATIVE_KEYWORDS = [
        "loss",
        "decline",
        "investigation",
        "fall",
        "negative",
        "weak",
        "layoffs",
        "lawsuit",
        "scandal",
        "bankruptcy",
    ]

    @classmethod
    def analyze_sentiment(cls, text: str) -> tuple[int, str, float]:
        text_lower = text.lower()

        positive_count = sum(1 for word in cls.POSITIVE_KEYWORDS if word in text_lower)
        negative_count = sum(1 for word in cls.NEGATIVE_KEYWORDS if word in text_lower)

        # Determine signal based on keyword counts
        if positive_count > negative_count:
            signal = TradeSignalEnum.BUY
            justification = (
                f"Fallback analysis detected {positive_count} positive indicators"
            )
        elif negative_count > positive_count:
            signal = TradeSignalEnum.SELL
            justification = (
                f"Fallback analysis detected {negative_count} negative indicators"
            )
        else:
            signal = TradeSignalEnum.NEUTRAL
            justification = "Fallback analysis found mixed or unclear sentiment"

        # Low confidence for fallback analysis
        confidence = 0.1

        return int(signal), justification, confidence


@contract(
    pre_remedy=True,
    post_remedy=True,
    verbose=True,
    accumulate_errors=True,
)
class FinancialNewsAnalyzer(Expression):
    def __init__(self, config: Optional[ValidationConfig] = None):
        super().__init__()
        self.config = config or ValidationConfig()
        self.fallback_analyzer = FallbackAnalyzer()

    @property
    def prompt(self) -> str:
        return """You are an expert financial analyst specializing in news-driven trading signals.

Your task is to analyze financial news and generate trading recommendations based on:
1. Market sentiment analysis - How will investors react?
2. Company-specific impact assessment - Direct effects on the company
3. Industry trends and implications - Broader market context
4. Risk factors and market conditions - Uncertainty and volatility

Generate a trading signal:
- BUY (1): Strong positive news likely to drive stock price up
- NEUTRAL (0): Mixed or minimal impact on stock price  
- SELL (-1): Negative news likely to drive stock price down

Provide clear, specific justification referencing actual news content and realistic confidence levels.

Example JSON response:
{
  "trade_signal": 1,
  "justification": "Company reported 25% revenue growth and expanded into new markets, indicating strong fundamentals and growth trajectory",
  "confidence": 0.82
}"""

    def pre(self, input: NewsInput) -> bool:
        # Timestamp validation
        if not input.timestamp:
            raise ValueError("Timestamp cannot be empty")

        # News body validation
        if len(input.body.strip()) < self.config.MIN_NEWS_BODY_LENGTH:
            raise ValueError(
                f"News body must contain at least {self.config.MIN_NEWS_BODY_LENGTH} characters"
            )

        # Company name validation
        if len(input.company_name.strip()) < self.config.MIN_COMPANY_NAME_LENGTH:
            raise ValueError(
                f"Company name must be at least {self.config.MIN_COMPANY_NAME_LENGTH} characters"
            )

        logger.info(f"Input validation passed for {input.company_name}")
        return True

    def act(self, input: NewsInput, **kwargs) -> NewsInput:
        cleaned_and_normalized_input_body = " ".join(input.body.strip().split())
        normalized_company_name = input.company_name.strip().title()

        logger.info(f"Preprocessed input for {normalized_company_name}")

        return NewsInput(
            timestamp=input.timestamp,
            body=cleaned_and_normalized_input_body,
            company_name=normalized_company_name,
        )

    def post(self, output: TradingSignal) -> bool:
        if output.trade_signal not in [-1, 0, 1]:
            raise ValueError("Trade signal must be exactly -1, 0, or 1")

        if len(output.justification.strip()) < self.config.MIN_JUSTIFICATION_LENGTH:
            raise ValueError(
                f"Justification must be at least {self.config.MIN_JUSTIFICATION_LENGTH} characters"
            )

        # Business logic validation
        if (
            abs(output.trade_signal) == 1
            and output.confidence < self.config.MIN_CONFIDENCE_FOR_STRONG_SIGNAL
        ):
            raise ValueError(
                f"Strong signals require confidence >= {self.config.MIN_CONFIDENCE_FOR_STRONG_SIGNAL}"
            )

        if (
            output.trade_signal == 0
            and output.confidence > self.config.MAX_CONFIDENCE_FOR_NEUTRAL
        ):
            raise ValueError(
                f"Neutral signals with confidence > {self.config.MAX_CONFIDENCE_FOR_NEUTRAL} seem inconsistent"
            )

        logger.info(
            f"Output validation passed: {TradeSignalEnum(output.trade_signal).name}"
        )
        return True

    def forward(self, input: NewsInput, **kwargs) -> TradingSignal:
        if self.contract_successful:
            logger.info(f"✅ Contract validation successful for {input.company_name}")
            return self.contract_result
        else:
            logger.warning(
                f"⚠️ Contract validation failed for {input.company_name}, using fallback"
            )

            signal, justification, confidence = (
                self.fallback_analyzer.analyze_sentiment(input.body)
            )

            return TradingSignal(
                trade_signal=signal,
                justification=justification,
                confidence=confidence,
            )


class NewsAnalysisProcessor:
    def __init__(self, config: Optional[ValidationConfig] = None):
        self.analyzer = FinancialNewsAnalyzer(config)
        self.results: List[Dict[str, Any]] = []

    def process_single_news(self, news_data: Dict[str, str]) -> Dict[str, Any]:
        # Create and validate input model
        news_input = NewsInput(**news_data)

        # Process with contract
        result = self.analyzer(input=news_input)

        # Convert to output dictionary
        output = {
            "company_name": news_data["company_name"],
            "trade_signal": result.trade_signal,
            "signal_name": TradeSignalEnum(result.trade_signal).name,
            "justification": result.justification,
            "confidence": result.confidence,
            "timestamp": news_data["timestamp"],
            "status": "success",
        }

        logger.info(
            f"Successfully processed {news_data['company_name']}: {output['signal_name']} ({output['confidence']:.2f})"
        )
        return output

    def process_batch_news(
        self, news_list: List[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        logger.info(f"🔍 Processing {len(news_list)} news articles...")

        results = []
        for i, news_data in enumerate(news_list, 1):
            single_news_result = self.process_single_news(news_data)
            results.append(single_news_result)

        self.results = results
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

    processor = NewsAnalysisProcessor()
    results = processor.process_batch_news(example_inputs)

    print("\n📊 FINAL RESULTS SUMMARY")
    print("=" * 50)

    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['company_name']}")
        print(f"   Signal: {result['signal_name']} ({result['trade_signal']})")
        print(f"   Confidence: {result['confidence']:.2f}")
        print(f"   Justification: {result['justification']}")


if __name__ == "__main__":
    main()
