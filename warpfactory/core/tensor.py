"""
Tensor data structure for WarpFactory

This module provides the Tensor class that mimics MATLAB's struct-based
tensor representation but uses Python dictionaries and numpy arrays.
"""

import numpy as np
from typing import Dict, List, Literal, Optional
from datetime import date


class Tensor:
    """
    A tensor representation for spacetime metrics and stress-energy tensors.

    The tensor is stored as a dictionary of 4x4 components, where each component
    is a 4D numpy array representing values across the spacetime grid (t, x, y, z).

    Attributes:
        tensor (Dict): 4x4 dictionary of numpy arrays
        type (str): Either "metric" or "stress-energy"
        name (str): Name of the tensor (e.g., "Alcubierre", "Minkowski")
        index (str): Index position - "covariant", "contravariant", "mixedupdown", "mixeddownup"
        coords (str): Coordinate system (currently only "cartesian" supported)
        scaling (List[float]): Grid scaling factors [t, x, y, z]
        params (Dict): Dictionary of parameters used to create the tensor
        date (str): Date of creation
    """

    def __init__(
        self,
        tensor: Dict[tuple, np.ndarray],
        tensor_type: Literal["metric", "stress-energy"],
        name: str = "",
        index: Literal["covariant", "contravariant", "mixedupdown", "mixeddownup"] = "covariant",
        coords: str = "cartesian",
        scaling: Optional[List[float]] = None,
        params: Optional[Dict] = None,
    ):
        """
        Initialize a Tensor object.

        Args:
            tensor: Dictionary with (i,j) tuples as keys and 4D arrays as values
            tensor_type: Type of tensor ("metric" or "stress-energy")
            name: Name identifier for the tensor
            index: Index position type
            coords: Coordinate system
            scaling: Grid scaling factors [t, x, y, z]
            params: Parameters dictionary
        """
        self.tensor = tensor
        self.type = tensor_type
        self.name = name
        self.index = index
        self.coords = coords
        self.scaling = scaling if scaling is not None else [1.0, 1.0, 1.0, 1.0]
        self.params = params if params is not None else {}
        self.date = str(date.today())

    @property
    def shape(self):
        """Get the shape of the spacetime grid"""
        return self.tensor[(0, 0)].shape

    def __getitem__(self, key: tuple) -> np.ndarray:
        """Access tensor components using (i, j) indexing"""
        return self.tensor[key]

    def __setitem__(self, key: tuple, value: np.ndarray):
        """Set tensor components using (i, j) indexing"""
        self.tensor[key] = value

    def copy(self):
        """Create a deep copy of the tensor"""
        import copy
        return copy.deepcopy(self)

    def to_gpu(self):
        """
        Convert tensor arrays to GPU arrays using CuPy

        Returns:
            Tensor: New tensor with GPU arrays
        """
        try:
            import cupy as cp
            gpu_tensor = {}
            for key, value in self.tensor.items():
                gpu_tensor[key] = cp.asarray(value)

            result = self.copy()
            result.tensor = gpu_tensor
            return result
        except ImportError:
            raise ImportError("CuPy is not installed. Install with: pip install cupy")

    def to_cpu(self):
        """
        Convert tensor arrays from GPU to CPU using CuPy

        Returns:
            Tensor: New tensor with CPU arrays
        """
        try:
            import cupy as cp
            cpu_tensor = {}
            for key, value in self.tensor.items():
                if isinstance(value, cp.ndarray):
                    cpu_tensor[key] = cp.asnumpy(value)
                else:
                    cpu_tensor[key] = value

            result = self.copy()
            result.tensor = cpu_tensor
            return result
        except ImportError:
            return self.copy()

    def is_gpu(self) -> bool:
        """Check if tensor is stored on GPU"""
        try:
            import cupy as cp
            return isinstance(self.tensor[(0, 0)], cp.ndarray)
        except ImportError:
            return False

    @classmethod
    def zeros(cls, grid_size: List[int], **kwargs):
        """
        Create a zero tensor with specified grid size

        Args:
            grid_size: Shape of spacetime grid [t, x, y, z]
            **kwargs: Additional arguments for Tensor constructor

        Returns:
            Tensor: Zero-initialized tensor
        """
        tensor_dict = {}
        for i in range(4):
            for j in range(4):
                tensor_dict[(i, j)] = np.zeros(grid_size)

        return cls(tensor_dict, **kwargs)

    def __repr__(self):
        return (f"Tensor(type='{self.type}', name='{self.name}', "
                f"index='{self.index}', shape={self.shape})")
