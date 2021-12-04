import numpy as np
import random


class BaseTransform:
    
    def __init__(self, p):
        """
        p: Probability of applying transform.
        """

        assert 0 <= p <= 1
        self.p = p
        
    def apply(self, signals):
        raise NotImplementedError

    def __call__(self, signals):

        if random.random() < self.p:
            signals = self.apply(signals)
    
        return signals


class Compose:

    def __init__(self, transforms, p=1):
        """
        transforms: List of transforms to apply to signals.
        p: Probability of applying the list of transforms.
        """

        assert 0 <= p <= 1

        self.transforms = transforms
        self.p = p
        
    def __call__(self, signals):

        if random.random() < self.p:  
            for transform in self.transforms:
                signals = transform(signals)

        return signals


class FlipTime(BaseTransform):
    """Randomly flip signals along temporal dimension"""

    def __init__(self, p=0.5):
        """
        p: Probability of applying transform.
        """
        super().__init__(p)


    def apply(self, signals):

        if len(signals.shape) > 1:
            signals = np.fliplr(signals)

        else:
            signals = np.flipud(signals)

        return signals


class MaskTime(BaseTransform):
    """Randomly mask signal"""

    def __init__(self, min_fraction=0.0, max_fraction=0.5, p=0.5):
        """
        min_fraction: Minimum length of the mask as a fraction of the total time series length.
        max_fraction: Maximum length of the mask as a fraction of the total time series length.
        p: Probability of applying transform.
        """
        
        super().__init__(p)

        assert 0 <= min_fraction <= 1
        assert 0 <= max_fraction <= 1
        assert max_fraction >= min_fraction
        
        self.min_fraction = min_fraction
        self.max_fraction = max_fraction


    def apply(self, signals):

        num_samples = signals.shape[-1]
        length = random.randint(int(num_samples * self.min_fraction), int(num_samples * self.max_fraction))
        start = random.randint(0, num_samples - length)

        mask = np.zeros(length)
        masked_signals = signals.copy()
        masked_signals[..., start : start + length] *= mask

        return masked_signals


class Shift(BaseTransform):
    """Shift the signals forwards or backwards along the temporal dimension"""
    
    def __init__(self, min_fraction=-0.5, max_fraction=0.5, rollover=True, p=0.5):
        """
        min_fraction: Fraction of total timeseries to shift.
        max_fraction: Fraction of total timeseries to shift.
        rollover: Samples that roll beyond the first or last position are re-introduced at the last or first otherwise set to zero.
        p: Probability of applying this transform.
        """

        super().__init__(p)
        
        assert min_fraction >= -1
        assert max_fraction <= 1

        self.min_fraction = min_fraction
        self.max_fraction = max_fraction
        self.rollover = rollover

    def apply(self, signals):
        
        num_samples = signals.shape[-1]
        num_shift = int(round(random.uniform(self.min_fraction, self.max_fraction) * num_samples))
        signals = np.roll(signals, num_shift, axis=-1)

        if not self.rollover:
            if num_shift > 0:
                signals[..., :num_shift] = 0.0

            elif num_shift < 0:
                signals[..., num_shift:] = 0.0

        return signals


class FlipPolarity(BaseTransform):
    """Randomly flip sign of signal"""

    def __init__(self, p=0.5):
        """
        p: Probability of applying transform.
        """
        super().__init__(p)

    def apply(self, signals):
        return -signals


class GuassianNoise(BaseTransform):
    """Add gaussian noise to the signals"""

    def __init__(self, min_amplitude=0.001, max_amplitude=0.015, p=0.5):
        """
        min_amplitude: minimum amplitude of noise.
        max_amplitude: maximum amplitude of noise.
        p: Probability of applying this transform.
        """
        super().__init__(p)

        assert min_amplitude > 0.0
        assert max_amplitude > 0.0
        assert max_amplitude >= min_amplitude

        self.min_amplitude = min_amplitude
        self.max_amplitude = max_amplitude

    def apply(self, signals):

        amplitude = random.uniform(self.min_amplitude, self.max_amplitude)

        noise = np.random.randn(*signals.shape).astype(np.float32)
        signals = signals + amplitude * noise

        return signals