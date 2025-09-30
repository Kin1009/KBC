class MemoryErrorBase(Exception):
    """Base class for all memory allocator exceptions."""


class OutOfMemoryError(MemoryErrorBase):
    """Raised when malloc or realloc cannot allocate memory."""


class InvalidFreeError(MemoryErrorBase):
    """Raised when trying to free an invalid or unallocated pointer."""


class InvalidPointerError(MemoryErrorBase):
    """Raised when a pointer is not valid (not allocated)."""


class DataTooLargeError(MemoryErrorBase):
    """Raised when writing more data than the allocated block can hold."""
