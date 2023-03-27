import json
import re


def normalize_string(s: str) -> str:
    """Standardize strings to a specific format.

    Standardize the string by:
        - converting to lowercase,
        - trim, and
        - remove non alpha-numeric (except ,.!?) characters.

    Args:
        s: The string to standardize.

    Returns:
        A standardized version of the input string.
    """
    s = s.lower().strip()
    s = re.sub(r"([,.!?])", r" \1 ", s)  # isolate some marks
    s = re.sub(r"(['])", r"", s)  # remove apostrophe (i.e., shouldn't --> shouldnt)
    s = re.sub(
        r"[^a-zA-Z0-9,.!?]+", r" ", s
    )  # replace other characters with whitespace
    s = re.sub(r"\s+", r" ", s).strip()
    return s


class SubtitleWrapper:
    """Contains the subtitles for an audio file.

    Subtitles are expected to be contained in a JSON file with the format:
    {
        'alternative':
            []: # only the first element contains the following:
                {
                    'words': [],
                    <other data>
                }
    }
    where each element of 'words' contains a word and start and end times of the subtitle.

    Attributes:
        subtitle: A list of strings containing all the subtitles in order.
    """

    TIMESTAMP_PATTERN = re.compile("(\d+)?:?(\d{2}):(\d{2})[.,](\d{3})")

    def __init__(self, subtitle_path: str):
        self.subtitle = []
        self.load_gentle_subtitle(subtitle_path)

    def get(self) -> list:
        """Returns the subtitles as a list of words."""
        return self.subtitle

    def load_gentle_subtitle(self, subtitle_path: str) -> None:
        """Loads a single subtitle file into this object.

        Modifies the internal state of this object.
        The subtitles are loaded in order with a single word appended as a single element.
        Uses the gentle lib.

        Args:
            substitle_path: The string filepath to the subtitle (JSON) file.

        Raises:
            An exception if the specified file cannot be found.
        """
        try:
            with open(subtitle_path) as data_file:
                data = json.load(data_file)
                for item in data:
                    if "words" in item["alternatives"][0]:
                        raw_subtitle = item["alternatives"][0]["words"]
                        for word in raw_subtitle:
                            self.subtitle.append(word)
        except FileNotFoundError:
            self.subtitle = None

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
