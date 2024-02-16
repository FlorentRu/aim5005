import numpy as np
from typing import List, Tuple


class MinMaxScaler:
    def __init__(self):
        self.minimum = None
        self.maximum = None
        
    def _check_is_array(self, x: np.ndarray) -> np.ndarray:
        """
        Try to convert x to a np.ndarray if it's not a np.ndarray and return. If it can't be cast raise an error
        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)
            
        assert isinstance(x, np.ndarray), "Expected the input to be a numpy array"
        return x
        
    def fit(self, x: np.ndarray) -> None:   
        x = self._check_is_array(x)
        self.minimum = x.min(axis=0)
        self.maximum = x.max(axis=0)
        
    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        MinMax Scale the given vector
        """
        x = self._check_is_array(x)
        return (x - self.minimum) / (self.maximum - self.minimum)
    
    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        x = self._check_is_array(x)
        self.fit(x)
        return self.transform(x)
    
    
class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None
        
    def fit(self, x: np.ndarray) -> None:
        x = self._check_is_array(x)
        self.mean = np.mean(x, axis=0)
        self.std = np.std(x, axis=0)
        
    def transform(self, x: np.ndarray) -> np.ndarray:
        x = self._check_is_array(x)
        return (x - self.mean) / self.std
    
    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        x = self._check_is_array(x)
        self.fit(x)
        return self.transform(x)
    
    def _check_is_array(self, x: np.ndarray) -> np.ndarray:
        """
        Try to convert x to a np.ndarray if it's not a np.ndarray and return. If it can't be cast raise an error
        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)
            
        assert isinstance(x, np.ndarray), "Expected the input to be a numpy array"
        return x


def test_standard_scaler():
    scaler = StandardScaler() # Create a StandardScaler object
        
    data = np.array([[1, 2], [3, 4], [5, 6]]) # Create a sample data
        
    scaler.fit(data) # Fit and transform the data
    transformed_data = scaler.transform(data)
        
    expected_mean = np.array([3., 4.]) # Expected output
    expected_std = np.array([1.63299316, 1.63299316])
    expected_transformed_data = np.array([[-1.22474487, -1.22474487],
                                          [ 0.        ,  0.        ],
                                          [ 1.22474487,  1.22474487]])
    
    assert np.allclose(scaler.mean, expected_mean)     # Check if mean and std are calculated correctly
    assert np.allclose(scaler.std, expected_std)
    
    # Check if the transformed data matches the expected output
    assert np.allclose(transformed_data, expected_transformed_data)

