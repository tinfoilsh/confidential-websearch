"""Constants for the evals package."""

# Safeguard model configuration
SAFEGUARD_BASE_URL = "https://gpt-oss-safeguard-120b.inf5.tinfoil.sh"
SAFEGUARD_MODEL = "gpt-oss-safeguard-120b"

# Arbiter model configuration (for resolving classification disagreements)
ARBITER_BASE_URL = "https://kimi-k2-thinking.inf5.tinfoil.sh"
ARBITER_MODEL = "kimi-k2-thinking"

# Query generation model configuration
QUERY_GEN_BASE_URL = "https://gpt-oss-120b.inf5.tinfoil.sh"
QUERY_GEN_MODEL = "gpt-oss-120b"

# Content truncation limits
MAX_CONTENT_LENGTH = 500
