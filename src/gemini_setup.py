"""Gemini setup validation and configuration utilities.

This module provides utilities to validate Gemini API setup, including:
- Checking required package installations
- Validating API credentials
- Testing API connectivity
- Verifying model availability
"""
import os
import sys
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger("gemini_setup")

def check_dependencies() -> Dict[str, bool]:
    """Check if required Gemini packages are installed.
    
    Returns:
        Dict mapping package names to boolean indicating if installed
    """
    required_packages = {
        'google-generativeai': False,
        'google-auth': False,
        'numpy': False  # For vector operations
    }
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            required_packages[package] = True
        except ImportError:
            logger.error(f"Missing required package: {package}")
    
    return required_packages

def verify_credentials() -> bool:
    """Verify that necessary Gemini API credentials are configured.
    
    Returns:
        bool: True if credentials are properly configured
    """
    api_key = os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        logger.error("GOOGLE_API_KEY environment variable not set")
        return False
        
    # Could add additional credential validation here
    return True

def test_connection() -> bool:
    """Test connection to Gemini API.
    
    Returns:
        bool: True if connection test succeeds
    """
    try:
        import google.generativeai as genai
        genai.configure(api_key=os.environ.get('GOOGLE_API_KEY'))
        
        # Try listing models as a basic connectivity test
        _ = genai.list_models()
        return True
    except Exception as e:
        logger.error(f"Failed to connect to Gemini API: {e}")
        return False

def get_available_models() -> Dict[str, Any]:
    """Get information about available Gemini models.
    
    Returns:
        Dict containing model information
    """
    try:
        import google.generativeai as genai
        models = genai.list_models()
        return {
            model.name: {
                'supported_generation_methods': getattr(model, 'supported_generation_methods', []),
                'input_text_size': getattr(model, 'input_token_limit', 0),
                'description': getattr(model, 'description', '')
            }
            for model in models
        }
    except Exception as e:
        logger.error(f"Failed to get model information: {e}")
        return {}

def validate_setup() -> bool:
    """Run complete validation of Gemini setup.
    
    Checks:
    1. Required packages installed
    2. Credentials configured
    3. API connectivity
    4. Model availability
    
    Returns:
        bool: True if all validation steps pass
    """
    logger.info("Validating Gemini setup...")
    
    # Check dependencies
    deps = check_dependencies()
    if not all(deps.values()):
        missing = [pkg for pkg, installed in deps.items() if not installed]
        logger.error(f"Missing dependencies: {', '.join(missing)}")
        logger.error("Install with: pip install " + " ".join(missing))
        return False
        
    # Check credentials
    if not verify_credentials():
        logger.error("Credential verification failed")
        return False
        
    # Test connection
    if not test_connection():
        logger.error("Connection test failed")
        return False
        
    # Check model availability
    models = get_available_models()
    if not models:
        logger.error("No models available")
        return False
        
    logger.info("Gemini setup validation successful!")
    logger.info(f"Available models: {list(models.keys())}")
    return True

if __name__ == '__main__':
    # Set up logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
    
    # Run validation
    if validate_setup():
        print("Gemini setup validated successfully!")
        sys.exit(0)
    else:
        print("Gemini setup validation failed!")
        sys.exit(1)