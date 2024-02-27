import numpy as np

class Spectrum:
    def __init__(self, wavelengths: list[float], amplitudes: list[float]):
        if len(wavelengths) != len(amplitudes):
            raise SystemError("Length of wavelength and amplitude vectors must match.")
        self.wavelengths = np.array(wavelengths, float)
        positive = np.all(self.wavelengths > 0)
        sorted_ascending = (np.diff(self.wavelengths) >= 0).all()
        if not positive or not sorted_ascending:
            raise SystemError("Spectral wavelengths must be a vector of unique values in ascending order.")
        self.amplitudes = np.array(amplitudes, float)

    def __len__(self) -> int:
        return self.wavelengths.size
    
    def __getitem__(self, index) -> tuple:
        return (self.wavelengths[index], self.amplitudes[index])

    def __iter__(self):
        pass

    def __next__(self):
        pass



s = Spectrum([-1,2.99,3],[4,5,6])
print(str(s[0]))
print(len(s))