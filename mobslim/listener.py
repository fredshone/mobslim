import pandas as pd


class EventListener:
    """
    Base class for event listeners in the simulation.
    """

    def __init__(self):
        self.log = []

    def add(self, event, idx, time, uv) -> None:
        self.log.append((event, idx, time, uv))

    def reset(self) -> None:
        """Reset the event listener state."""
        self.log = []


class CSVChunkWriter:
    """
    Extend a list of lines (dicts) that are saved to drive once they reach a certain length.
    """

    def __init__(self, path, compression=None, chunksize=1000) -> None:
        self.path = path
        self.compression = compression
        self.chunksize = chunksize

        self.chunk = []
        self.idx = 0

    def add(self, lines: list) -> None:
        """
        Add a list of lines (dicts) to the chunk.
        If chunk exceeds chunksize, then write to disk.
        :param lines: list of dicts
        :return: None
        """
        self.chunk.extend(lines)
        if len(self.chunk) > self.chunksize:
            self.write()

    def write(self) -> None:
        """
        Convert chunk to dataframe and write to disk.
        :return: None
        """
        chunk_df = pd.DataFrame(
            self.chunk, index=range(self.idx, self.idx + len(self.chunk))
        )
        if not self.idx:
            chunk_df.to_csv(self.path, compression=self.compression)
            self.idx += len(self.chunk)
        else:
            chunk_df.to_csv(
                self.path, header=None, mode="a", compression=self.compression
            )
            self.idx += len(self.chunk)
        del chunk_df
        self.chunk = []

    def finish(self) -> None:
        self.write()
        self.logger.info(f"Chunkwriter finished for {self.path}")

    def __len__(self):
        return self.idx + len(self.chunk)
