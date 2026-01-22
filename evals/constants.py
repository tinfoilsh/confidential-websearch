"""Constants for the evals package."""

# Base URL for all Tinfoil models
TINFOIL_BASE_URL = "https://inference.tinfoil.sh"

# Safeguard model configuration
SAFEGUARD_BASE_URL = TINFOIL_BASE_URL
SAFEGUARD_MODEL = "gpt-oss-safeguard-120b"

# Labeler model configuration (for independent ground truth labeling)
LABELER_BASE_URL = TINFOIL_BASE_URL
LABELER_MODEL = "qwen3-coder-480b"

# Query generation model configuration
QUERY_GEN_BASE_URL = TINFOIL_BASE_URL
QUERY_GEN_MODEL = "gpt-oss-120b"

# Content truncation limits
MAX_CONTENT_LENGTH = 500
