import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
import atexit
import shutil
import glob

# Global variable to store the log file path
_log_file = None

def archive_old_logs():
    """Archive old log files into a dated directory."""
    log_dir = "logs"
    archive_dir = os.path.join(log_dir, "archive")
    
    # Create archive directory if it doesn't exist
    os.makedirs(archive_dir, exist_ok=True)
    
    # Get current date for archive folder
    current_date = datetime.now().strftime("%Y%m%d")
    date_archive_dir = os.path.join(archive_dir, current_date)
    os.makedirs(date_archive_dir, exist_ok=True)
    
    # Move all existing log files to archive
    for log_file in glob.glob(os.path.join(log_dir, "classify_*.log*")):
        if os.path.isfile(log_file):
            # Skip if file is currently being written to
            if log_file == _log_file:
                continue
            
            # Move file to dated archive directory
            try:
                shutil.move(log_file, os.path.join(date_archive_dir, os.path.basename(log_file)))
            except (shutil.Error, OSError) as e:
                logging.warning(f"Failed to archive log file {log_file}: {e}")

def get_log_file():
    """Get the current log file path."""
    global _log_file
    if _log_file is None:
        # Create logs directory if it doesn't exist
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        
        # Archive old logs before creating new one
        archive_old_logs()
        
        # Create a timestamp-based log filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        _log_file = os.path.join(log_dir, f"classify_{timestamp}.log")
    
    return _log_file

def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure logging settings with both console and file output."""
    # Get or create the log file path
    log_file = get_log_file()
    
    # Configure formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Configure file handler with rotation (max 10MB per file, keep 5 backup files)
    file_handler = RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"  # 10MB
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    
    # Configure console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    
    # Configure root logger
    root_logger = logging.getLogger()
    if not root_logger.handlers:  # Only add handlers if they don't exist
        root_logger.setLevel(level)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
    
    # Set PIL logging to INFO to suppress debug messages
    logging.getLogger("PIL").setLevel(logging.INFO)
    
    # Get our logger
    logger = logging.getLogger(__name__)
    if not logger.handlers:  # Log initialization only once
        logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for the given name."""
    return logging.getLogger(name)

# Clean up function to close log handlers
def cleanup():
    """Close all log handlers to ensure proper file cleanup."""
    for handler in logging.getLogger().handlers:
        handler.close()

# Register cleanup function
atexit.register(cleanup)
