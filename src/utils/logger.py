"""
Logging configuration for AI Beauty Platform
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from typing import Optional

def setup_logger(
    name: str = 'ai_beauty_platform',
    level: str = 'INFO',
    log_file: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Set up a comprehensive logger for the application
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional log file path
        max_bytes: Maximum log file size before rotation
        backup_count: Number of backup log files to keep
        
    Returns:
        Configured logger instance
    """
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Console handler with simple format
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    # File handler with detailed format (if log_file is specified)
    if log_file:
        # Ensure log directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        # Rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
    
    # Error handler for critical errors
    error_handler = logging.StreamHandler(sys.stderr)
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    logger.addHandler(error_handler)
    
    return logger

def get_logger(name: str = None) -> logging.Logger:
    """
    Get a logger instance
    
    Args:
        name: Logger name (optional)
        
    Returns:
        Logger instance
    """
    if name is None:
        name = 'ai_beauty_platform'
    
    return logging.getLogger(name)

class PerformanceLogger:
    """Context manager for logging performance metrics"""
    
    def __init__(self, logger: logging.Logger, operation: str):
        self.logger = logger
        self.operation = operation
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.info(f"Starting {self.operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = (datetime.now() - self.start_time).total_seconds()
            if exc_type is None:
                self.logger.info(f"Completed {self.operation} in {duration:.3f}s")
            else:
                self.logger.error(f"Failed {self.operation} after {duration:.3f}s: {exc_val}")
    
    def log_progress(self, message: str, **kwargs):
        """Log progress message with optional context"""
        context = ' '.join([f"{k}={v}" for k, v in kwargs.items()])
        if context:
            self.logger.info(f"{self.operation}: {message} - {context}")
        else:
            self.logger.info(f"{self.operation}: {message}")

def log_function_call(logger: logging.Logger):
    """Decorator to log function calls with parameters and timing"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Log function call
            params = []
            if args:
                params.append(f"args={args}")
            if kwargs:
                params.append(f"kwargs={kwargs}")
            
            param_str = ", ".join(params) if params else "no parameters"
            logger.debug(f"Calling {func.__name__}({param_str})")
            
            # Execute function and log result
            start_time = datetime.now()
            try:
                result = func(*args, **kwargs)
                duration = (datetime.now() - start_time).total_seconds()
                logger.debug(f"Completed {func.__name__} in {duration:.3f}s")
                return result
            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds()
                logger.error(f"Failed {func.__name__} after {duration:.3f}s: {e}")
                raise
        
        return wrapper
    return decorator

# Create default logger
default_logger = setup_logger()

if __name__ == "__main__":
    # Test logging setup
    logger = setup_logger('test_logger', 'DEBUG', 'logs/test.log')
    
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    
    # Test performance logger
    with PerformanceLogger(logger, "test_operation"):
        import time
        time.sleep(0.1)
        logger.info("Processing...")
    
    # Test function decorator
    @log_function_call(logger)
    def test_function(x, y=10):
        return x + y
    
    result = test_function(5, y=15)
    logger.info(f"Function result: {result}")
