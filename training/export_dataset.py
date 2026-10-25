"""Export the generators' training data: aligned daily log returns, to npz.

Run locally (light pandas, no training):

    uv run python training/export_dataset.py OUT.npz

Reads the portfolio page's adjusted closes (``data/portfolio/adjusted.parquet``,
read-only). Six price series, each in its own currency: the three US stocks,
the S&P 500 price index, KOSPI and USD/KRW. ``^IRX`` is a yield, not a price,
and ``^SP500TR`` all but duplicates ``^GSPC``; both are left out.

Mixed calendars (Seoul and New York): only days on which all six have a close
are kept (inner join), so a return spans each market's holidays instead of
showing a carried-over zero. Each return is the log change between
consecutive kept days. The same export (``finagent.colab_jobs.export``) runs
for the app's 설정 > 생성 모델 학습 with the chosen tickers and dates.
"""

# ruff: noqa: T201, INP001

from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

import pandas as pd

from finagent.colab_jobs import export

SOURCE: Final = Path("data") / "portfolio" / "adjusted.parquet"
TICKERS: Final = ("AAPL", "MSFT", "PGR", "^GSPC", "^KS11", "KRW=X")


def main(out: Path) -> None:
    rets = export(out, pd.read_parquet(SOURCE), TICKERS)
    first, last = (d.strftime("%Y-%m-%d") for d in rets.index[[0, -1]])
    print(f"{out}: {rets.shape[0]} days x {rets.shape[1]}, {first}..{last}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
