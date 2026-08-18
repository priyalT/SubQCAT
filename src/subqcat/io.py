from pathlib import Path

import dask.dataframe as dd


class XeniumBundle:
    def __init__(self, path: Path):
        self.path = Path(path)

    def validate_transcripts(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(f"Xenium output bundle not found: {self.path}")
        if not self.path.is_dir():
            raise ValueError(f"Expected a Xenium output bundle directory: {self.path}")
        transcript_file = self.path / "transcripts.parquet"
        if not transcript_file.exists():
            raise FileNotFoundError(f"transcripts.parquet not found in Xenium bundle: {self.path}")

    def validate_cells(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(f"Xenium output bundle not found: {self.path}")
        if not self.path.is_dir():
            raise ValueError(f"Expected a Xenium output bundle directory: {self.path}")
        cells_file = self.path / "cells.parquet"
        if not cells_file.exists():
            raise FileNotFoundError(f"cells.parquet not found in Xenium bundle: {self.path}")

    @property
    def cells_file(self) -> Path:
        return self.path / "cells.parquet"

    @property
    def transcript_file(self) -> Path:
        return self.path / "transcripts.parquet"

    def load_transcripts_dataframe(self) -> dd.DataFrame:
        self.validate_transcripts()
        try:
            return dd.read_parquet(self.transcript_file)
        except Exception as e:
            raise ValueError(f"Unable to read {self.transcript_file} as a parquet file.") from e

    def load_cells_dataframe(self) -> dd.DataFrame:
        self.validate_cells()
        try:
            return dd.read_parquet(self.cells_file)
        except Exception as e:
            raise ValueError(f"Unable to read {self.cells_file} as a parquet file.") from e

