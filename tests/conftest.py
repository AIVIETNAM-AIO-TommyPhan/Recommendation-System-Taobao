import pandas as pd
import pytest


@pytest.fixture
def ingested_at() -> pd.Timestamp:
    return pd.Timestamp("2024-01-01T00:00:00Z")
