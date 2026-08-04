import random

DOMAINS = {
    "customer_churn": {
        "label": "customer churn prediction",
        "dataset": ["customer churn dataset", "subscription churn dataset", "customer retention dataset"],
        "target": ["customer churn", "subscription cancellation", "customer attrition"],
        "model": ["logistic regression model", "gradient boosting model", "random forest classifier"],
        "feature": ["tenure", "monthly charges", "contract type", "usage frequency"],
    },
    "fraud_detection": {
        "label": "fraud detection",
        "dataset": ["transaction dataset", "credit card transactions dataset", "payments dataset"],
        "target": ["fraudulent transactions", "anomalous payments", "fraud risk"],
        "model": ["isolation forest model", "gradient boosting model", "autoencoder model"],
        "feature": ["transaction amount", "merchant category", "transaction frequency", "device fingerprint"],
    },
    "recommendation_system": {
        "label": "product recommendation",
        "dataset": ["user interaction dataset", "clickstream dataset", "purchase history dataset"],
        "target": ["user click-through rate", "purchase likelihood", "item relevance"],
        "model": ["collaborative filtering model", "matrix factorisation model", "two-tower neural network"],
        "feature": ["item embeddings", "user embeddings", "session length", "category preference"],
    },
    "sentiment_analysis": {
        "label": "customer sentiment analysis",
        "dataset": ["product review dataset", "customer feedback dataset", "social media comments dataset"],
        "target": ["review sentiment", "customer satisfaction", "complaint category"],
        "model": ["fine-tuned transformer model", "LSTM classifier", "naive Bayes classifier"],
        "feature": ["review length", "word embeddings", "star rating", "topic distribution"],
    },
    "image_classification": {
        "label": "product image classification",
        "dataset": ["product image dataset", "defect inspection image dataset", "catalogue image dataset"],
        "target": ["product category", "defect presence", "image quality label"],
        "model": ["convolutional neural network", "fine-tuned ResNet model", "vision transformer model"],
        "feature": ["pixel intensity", "image resolution", "colour histogram", "edge features"],
    },
    "sales_forecasting": {
        "label": "sales demand forecasting",
        "dataset": ["historical sales dataset", "store-level sales dataset", "demand forecasting dataset"],
        "target": ["weekly sales volume", "product demand", "inventory requirements"],
        "model": ["gradient boosting regressor", "SARIMA model", "temporal fusion transformer"],
        "feature": ["seasonality", "promotional calendar", "store location", "lag features"],
    },
    "predictive_maintenance": {
        "label": "equipment predictive maintenance",
        "dataset": ["sensor telemetry dataset", "equipment logs dataset", "maintenance records dataset"],
        "target": ["equipment failure", "remaining useful life", "maintenance urgency"],
        "model": ["random forest classifier", "gradient boosting model", "LSTM survival model"],
        "feature": ["vibration readings", "temperature readings", "operating hours", "sensor drift"],
    },
    "credit_scoring": {
        "label": "credit risk scoring",
        "dataset": ["loan applications dataset", "credit bureau dataset", "borrower repayment dataset"],
        "target": ["loan default risk", "creditworthiness", "repayment probability"],
        "model": ["logistic regression model", "XGBoost model", "scorecard model"],
        "feature": ["credit utilisation", "income-to-debt ratio", "repayment history", "loan term"],
    },
    "supply_chain_optimisation": {
        "label": "supply chain optimisation",
        "dataset": ["logistics dataset", "warehouse inventory dataset", "shipment tracking dataset"],
        "target": ["delivery delay", "stock-out risk", "route efficiency"],
        "model": ["gradient boosting regressor", "mixed-integer optimisation model", "random forest regressor"],
        "feature": ["lead time", "warehouse capacity", "route distance", "supplier reliability"],
    },
    "healthcare_readmission": {
        "label": "patient readmission risk",
        "dataset": ["patient records dataset", "hospital admissions dataset", "clinical notes dataset"],
        "target": ["30-day readmission", "patient risk score", "length of stay"],
        "model": ["gradient boosting model", "logistic regression model", "survival analysis model"],
        "feature": ["comorbidity count", "prior admissions", "lab results", "discharge disposition"],
    },
    "energy_consumption": {
        "label": "energy consumption forecasting",
        "dataset": ["smart meter dataset", "building energy dataset", "grid load dataset"],
        "target": ["hourly energy consumption", "peak load", "energy anomalies"],
        "model": ["gradient boosting regressor", "LSTM forecasting model", "prophet model"],
        "feature": ["weather data", "occupancy patterns", "time of day", "historical load"],
    },
    "text_summarisation": {
        "label": "document summarisation",
        "dataset": ["internal reports dataset", "news articles dataset", "support tickets dataset"],
        "target": ["summary quality", "key topic extraction", "document category"],
        "model": ["fine-tuned summarisation model", "extractive summarisation pipeline", "topic model"],
        "feature": ["document length", "sentence embeddings", "keyword frequency", "topic coherence"],
    },
    "ab_testing_analysis": {
        "label": "product experimentation analysis",
        "dataset": ["A/B test results dataset", "experiment exposure dataset", "conversion funnel dataset"],
        "target": ["conversion rate uplift", "statistical significance", "experiment outcome"],
        "model": ["Bayesian A/B testing model", "causal inference model", "difference-in-differences model"],
        "feature": ["exposure group", "conversion event", "sample size", "variance reduction covariate"],
    },
    "network_intrusion_detection": {
        "label": "network intrusion detection",
        "dataset": ["network traffic dataset", "intrusion detection dataset", "firewall logs dataset"],
        "target": ["malicious traffic", "intrusion type", "anomaly score"],
        "model": ["random forest classifier", "autoencoder model", "gradient boosting model"],
        "feature": ["packet size", "connection duration", "protocol type", "byte rate"],
    },
    "employee_attrition": {
        "label": "employee attrition prediction",
        "dataset": ["HR records dataset", "employee survey dataset", "workforce analytics dataset"],
        "target": ["employee attrition", "engagement score", "resignation risk"],
        "model": ["logistic regression model", "gradient boosting model", "survival analysis model"],
        "feature": ["tenure", "performance rating", "compensation band", "manager changes"],
    },
}

DOMAIN_KEYS = list(DOMAINS.keys())


def pick_domain_vocab(domain_key: str, rng: random.Random) -> dict:
    """Return a dict of concrete placeholder values sampled for one task instance."""
    d = DOMAINS[domain_key]
    return {
        "dataset": rng.choice(d["dataset"]),
        "target": rng.choice(d["target"]),
        "model": rng.choice(d["model"]),
        "feature": rng.choice(d["feature"]),
        "domain_label": d["label"],
    }
