"""
Error logging utility for tracking and storing error messages.
"""

import logging
import os
from datetime import datetime
from typing import Optional

class ErrorLogger:
    """Custom logger for error tracking."""
    
    def __init__(self, log_dir: str = "logs", filename: Optional[str] = None):
        """Initialize error logger.
        
        Args:
            log_dir: Directory to store error logs
            filename: Optional specific filename for the error log
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Set up logger
        self.logger = logging.getLogger('error_logger')
        self.logger.setLevel(logging.ERROR)
        
        # Remove any existing handlers to avoid duplicate logging
        self.logger.handlers = []
        
        # Create file handler
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'error_log_{timestamp}.log'
        
        file_handler = logging.FileHandler(
            os.path.join(log_dir, filename),
            mode='a'
        )
        file_handler.setLevel(logging.ERROR)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.ERROR)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def log_error(self, error: Exception, context: str = ""):
        """Log an error with optional context information.
        
        Args:
            error: The exception to log
            context: Optional context about where/why the error occurred
        """
        error_message = f"{context} - {str(error)}" if context else str(error)
        self.logger.error(error_message, exc_info=True)
    
    def log_error_msg(self, message: str):
        """Log an error message directly.
        
        Args:
            message: The error message to log
        """
        self.logger.error(message)
    
    def error(self, msg: str) -> None:
        """Log an error message."""
        self.logger.error(msg)
    
    def exception(self, msg: str) -> None:
        """Log an exception with traceback."""
        self.logger.exception(msg)

# Global error logger instance
_error_logger = None

def get_error_logger(log_dir: str = "logs", filename: Optional[str] = None) -> ErrorLogger:
    """Get or create the global error logger instance.
    
    Args:
        log_dir: Directory to store error logs
        filename: Optional specific filename for the error log
    
    Returns:
        ErrorLogger instance
    """
    global _error_logger
    if _error_logger is None:
        _error_logger = ErrorLogger(log_dir, filename)
    return _error_logger
