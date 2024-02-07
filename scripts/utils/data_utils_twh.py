"""Utility file to read TSV files containing subtitles into an object.

Typical usage example:
    s = SubtitleWrapper('dataset/subtitles.tsv')
    words = s.get()
"""

import re


def normalize_string(s: str) -> str:
    """Standardize strings to a specific format.

    Standardize the string by:
        - converting to lowercase,
        - trim, and
        - remove non alpha-numeric characters.

    Args:
        s: The string to standardize.

    Returns:
        A standardized version of the input string.
    """
    s = s.lower().strip()
    # s = re.sub(r"([,.!?])", r" \1 ", s)  # isolate some marks
    s = re.sub(r"([,.!?])", r"", s)  # remove marks
    s = re.sub(r"(['])", r"", s)  # remove apostrophe (i.e., shouldn't --> shouldnt)
    s = re.sub(
        r"[^a-zA-Z0-9,.!?]+", r" ", s
    )  # replace other characters with whitespace
    s = re.sub(r"\s+", r" ", s).strip()
    return s


class SubtitleWrapper:
    """Contains the subtitles converted from a JSON file.

    Subtitles are expected to be contained in a TSV file with each line containing words.

    Attributes:
        subtitle: A list of strings containing all the subtitles in order.
    """

    TIMESTAMP_PATTERN = re.compile("(\d+)?:?(\d{2}):(\d{2})[.,](\d{3})")

    def __init__(self, subtitle_path: str):
        """Initialization method.

        Args:
            subtitle_path: The string filepath to the subtitle (JSON) file.
        """
        self.subtitle = []
        self.load_tsv_subtitle(subtitle_path)

    def get(self) -> list:
        """Returns the subtitles as a list of words."""
        return self.subtitle

    def load_tsv_subtitle(self, subtitle_path: str) -> None:
        """Loads a single subtitle file into this object.

        Modifies the internal state of this object.
        The subtitles are loaded in order with a single word appended as a single element.

        Args:
            substitle_path: The string filepath to the subtitle (TSV) file.

        Raises:
            An exception if the specified file cannot be found.
        """
        try:
            with open(subtitle_path) as file:
                # I had to update it since file number 157 has a different structure and make some problems
                for line in file:
                    line: str = line.strip()
                    if line.__contains__("content\t\t"):
                        line = line[len("content\t\t") :]
                        print("*****************************Error Content")
                    splitted = line.split("\t")
                    if len(splitted) == 2:
                        splitted.append("eh")
                        print(
                            "*****************************Error Lost Word "
                            + splitted[0]
                        )
                    self.subtitle.append(splitted)

        except FileNotFoundError:
            self.subtitle = None
        print()

    def get_seconds(self, word_time_e: str) -> float:
        """Convert a timestamp into seconds.

        Args:
            word_time_e: The timestamp as a string (ex. hrs:mins:secs.milli - 02:02:02.125).

        Returns:
            The timestamp as a float seconds starting from zero.
        """
        time_value = re.match(self.TIMESTAMP_PATTERN, word_time_e)
        if not time_value:
            print("wrong time stamp pattern")
            exit()

        values = list(map(lambda x: int(x) if x else 0, time_value.groups()))
        hours, minutes, seconds, milliseconds = (
            values[0],
            values[1],
            values[2],
            values[3],
        )

        return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000
