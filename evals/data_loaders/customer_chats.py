"""Data loader for customer service chat conversations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class ChatTurn:
    """A single turn in a conversation."""

    role: str  # "user" or "assistant"
    content: str


@dataclass
class Conversation:
    """A customer service conversation."""

    id: str
    language: str
    turns: list[ChatTurn]
    file: str


def load_customer_chats(
    dataset_dir: str | Path,
    max_conversations: int | None = None,
) -> list[Conversation]:
    """
    Load customer service conversations from XLSX files.

    Expects directory structure like:
        dataset_dir/
        ├── Batch 1/
        │   ├── English US/
        │   │   └── chat1.xlsx
        │   ├── German/
        │   └── ...
        └── Batch 2/
            └── ...

    Each XLSX file contains a two-column conversation where:
    - Column headers identify the first speaker
    - Rows contain alternating turns

    Args:
        dataset_dir: Path to the dataset directory
        max_conversations: Maximum conversations to load (None for all)

    Returns:
        List of Conversation objects
    """
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    conversations = []
    xlsx_files = list(dataset_path.rglob("*.xlsx"))
    xlsx_files = [f for f in xlsx_files if "__MACOSX" not in str(f)]

    for xlsx_path in sorted(xlsx_files):
        if max_conversations is not None and len(conversations) >= max_conversations:
            break

        try:
            df = pd.read_excel(xlsx_path)
            if len(df.columns) < 2:
                continue

            # First turn comes from column headers
            first_role_raw = str(df.columns[0]).strip().rstrip(":")
            first_content = str(df.columns[1])

            if any(r in first_role_raw for r in ["Customer", "Client", "User"]):
                first_role = "user"
            else:
                first_role = "assistant"

            turns = [ChatTurn(role=first_role, content=first_content)]

            # Remaining turns from rows
            for _, row in df.iterrows():
                if not pd.notna(row.iloc[0]) or not pd.notna(row.iloc[1]):
                    continue
                role_raw = str(row.iloc[0]).strip().rstrip(":")
                content = str(row.iloc[1])
                if any(r in role_raw for r in ["Customer", "Client", "User"]):
                    chat_role = "user"
                else:
                    chat_role = "assistant"
                turns.append(ChatTurn(role=chat_role, content=content))

            # Extract language from path
            language = "unknown"
            for part in xlsx_path.parts:
                if any(
                    lang in part
                    for lang in ["English", "German", "French", "Spanish"]
                ):
                    language = part
                    break

            conversations.append(
                Conversation(
                    id=xlsx_path.stem,
                    language=language,
                    turns=turns,
                    file=str(xlsx_path),
                )
            )
        except Exception as e:
            print(f"Error loading {xlsx_path}: {e}")

    return conversations


def get_user_turn_indices(conversation: Conversation) -> list[int]:
    """Get indices of all user turns in a conversation."""
    return [i for i, turn in enumerate(conversation.turns) if turn.role == "user"]
