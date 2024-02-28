import numpy as np

class Spectrum:
    def __init__(self, wavelengths: list[float], amplitudes: list[float]):
        """
        Radiometrc and photometric calculations assume amplitudes data in units of [uW/cm^2/nm] (absolute spectral irradiance).
        """
        if len(wavelengths) != len(amplitudes):
            raise SystemError("Length of wavelength and amplitude vectors must match.")
        
        self.wavelengths = np.array(wavelengths, float)

        positive = np.all(self.wavelengths > 0)
        sorted_ascending = (np.diff(self.wavelengths) >= 0).all()
        if not positive or not sorted_ascending:
            raise SystemError("Spectral wavelengths must be a vector of values in ascending order, where each value is greater than 0.")
        
        if self.wavelengths.size != np.unique(self.wavelengths).size:
            raise SystemError("All wavelength values must be unique.")

        self.amplitudes = np.array(amplitudes, float)

    def __len__(self) -> int:
        return self.wavelengths.size
    
    def __getitem__(self, index) -> tuple:
        return (self.wavelengths[index], self.amplitudes[index])

    def __str__(self) -> str:
        if len(self) == 1:
            return f"Spectrum(wavelength={self.wavelengths[0]}, amplitude={self.amplitudes[0]})"
        return f"Spectrum(wavelengths={self.wavelengths[0]}..{self.wavelengths[-1]}, amplitudes={self.amplitudes[0]}..{self.amplitudes[-1]})"

    def __iter__(self):
        pass

    def __next__(self):
        pass

    def getWavelengthIndices(self, lowerWavelength, upperWavelength) -> tuple[float, float]:
        """
        Returns indices associated with wavelength pair.
        """
        if lowerWavelength >= upperWavelength:
            raise SystemError("Lower wavelength argument must not be equal or larger than upper.")
        return (np.where(self.wavelengths == lowerWavelength)[0][0], np.where(self.wavelengths == upperWavelength)[0][0])



s = Spectrum([1,2.99,3],[4,5,6])
print(str(s[0]))
print(len(s))
print(str(s))
print(s.getWavelengthIndices(1,2.99))