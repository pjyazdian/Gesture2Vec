import csv
import json
import re


def normalize_string(s):
    """ lowercase, trim, and remove non-letter characters """
    s = s.lower().strip()
    # s = re.sub(r"([,.!?])", r" \1 ", s)  # isolate some marks
    s = re.sub(r"([,.!?])", r"", s)  # remove marks
    s = re.sub(r"(['])", r"", s)  # remove apostrophe (i.e., shouldn't --> shouldnt)
    s = re.sub(r"[^a-zA-Z0-9,.!?]+", r" ", s)  # replace other characters with whitespace
    s = re.sub(r"\s+", r" ", s).strip()
    return s


class SubtitleWrapper:
    TIMESTAMP_PATTERN = re.compile('(\d+)?:?(\d{2}):(\d{2})[.,](\d{3})')

    def __init__(self, subtitle_path):
        self.subtitle = []
        self.load_tsv_subtitle(subtitle_path)

    def get(self):
        return self.subtitle

    def load_tsv_subtitle(self, subtitle_path):
        try:
            with open(subtitle_path) as file:

                # I had to update it since file number 157 has a different structure and make some problems

                for line in file:
                    line = line.strip()
                    if line.__contains__("content\t\t"):
                        line = line[len('content\t\t'):]
                        print("*****************************Error Content")
                    splitted = line.split("\t")
                    if len(splitted)==2:
                        splitted.append("eh")
                        print("*****************************Error Lost Word " + splitted[0])

                    self.subtitle.append(splitted)

                # file.seek(0)
                # tsv_file = csv.reader(file, delimiter="\t")
                # for line in tsv_file:
                #     self.subtitle.append(line)


        except FileNotFoundError:
            self.subtitle = None
        print()

    # convert timestamp to second
    def get_seconds(self, word_time_e):
        time_value = re.match(self.TIMESTAMP_PATTERN, word_time_e)
        if not time_value:
            print('wrong time stamp pattern')
            exit()

        values = list(map(lambda x: int(x) if x else 0, time_value.groups()))
        hours, minutes, seconds, milliseconds = values[0], values[1], values[2], values[3]

        return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000
