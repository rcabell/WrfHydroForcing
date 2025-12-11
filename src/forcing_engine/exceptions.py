class FEException(Exception):
    """Custom exception for Forcing Engine errors."""
    pass

class FEConfigException(FEException):
    """Exception for configuration-related errors in the Forcing Engine."""
    pass