"""Class to hold average and current values (such as loss values).

Typical usage example:
    l = AverageMeter('autoencoder_loss')
    l.update(2.33)
"""

class AverageMeter(object):
    """Computes and stores the average and current value (such as loss values).

    Attributes:
        name: A string for the name for an instance of this object.
        fmt: A string that acts as a formatting value during printing (ex. :f).
        val: A float that is the most current value to be processed.
        avg: A float average value calculated using the most recent sum and count values.
        sum: A float running total that has been processed.
        count: An integer count of the number of times that val has been updated.
    """

    def __init__(self, name: str, fmt: str = ":f"):
        """Initialization method.

        Args:
            name: The string name for an instance of this object.
            fmt: A string formatting value during printing (ex. :f).
        """
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        """Reset all numerical attributes in this object.

        Modifies internal state of this object.
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        """Updates the numerical attributes in this object with the provided value and count.

        Modifies internal state of this object.

        Args:
            val: A float value to be used to update the calculations.
            n: A custom count value (default value is 1).
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self) -> str:
        """Print a custom formatted string with the val and avg values.

        Returns:
            The custom format string.
        """
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)
