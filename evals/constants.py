"""Constants for the evals package."""

# Base URL for all Tinfoil models
TINFOIL_BASE_URL = "https://inference.tinfoil.sh"

# Safeguard model configuration
SAFEGUARD_BASE_URL = TINFOIL_BASE_URL
SAFEGUARD_MODEL = "gpt-oss-safeguard-120b"

# DeepSeek labeler model configuration (for independent ground truth labeling)
DEEPSEEK_BASE_URL = TINFOIL_BASE_URL
DEEPSEEK_MODEL = "deepseek-r1-0528"

# Query generation model configuration
QUERY_GEN_BASE_URL = TINFOIL_BASE_URL
QUERY_GEN_MODEL = "gpt-oss-120b"

# Content truncation limits
MAX_CONTENT_LENGTH = 500
