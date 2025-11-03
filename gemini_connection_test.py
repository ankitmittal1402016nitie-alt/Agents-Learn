
import os
import logging
import sys
from dotenv import load_dotenv

# Set up basic logging with filename and line number
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)

def run_test():
    """A minimal test to check for Gemini SDK import and connection."""
    logger.info("--- Starting Gemini Connection Test ---")

    # 1. Load environment variables
    try:
        load_dotenv()
        logger.info("Attempted to load .env file.")
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key or "your-actual-api-key" in api_key:
            logger.warning("GOOGLE_API_KEY is not set or is using a placeholder value.")
            # We can still proceed to test the import
        else:
            logger.info("GOOGLE_API_KEY is loaded.")
    except Exception as e:
        logger.error(f"Failed to load dotenv: {e}")
        # Continue anyway to test the import

    # 2. Test the import
    try:
        import google.generativeai as genai
        logger.info("Successfully imported 'google.generativeai'.")
    except ImportError as e:
        logger.error(f"Failed to import 'google.generativeai'. Error: {e}")
        logger.error("This indicates a problem with the Python environment or package installation.")
        logger.error(f"Python executable: {sys.executable}")
        logger.error(f"Python path: {sys.path}")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred during import: {e}")
        return False

    # 3. Test API configuration and connection (only if API key is present)
    if api_key and "your-actual-api-key" not in api_key:
        try:
            genai.configure(api_key=api_key)
            logger.info("Successfully configured Gemini with API key.")
            
            # 4. List models as a basic API call test
            logger.info("Attempting to list available models...")
            models = list(genai.list_models())
            if models:
                logger.info(f"Successfully listed {len(models)} models.")
                # for m in models:
                #     logger.info(f" - Model: {m.name}")
            else:
                logger.warning("API call succeeded but no models were returned.")
            logger.info("--- Connection Test Successful ---")
            return True
        except Exception as e:
            logger.error(f"An error occurred during API configuration or connection: {e}")
            logger.error("This could be due to an invalid API key, network issues, or permissions problems.")
            return False
    else:
        logger.warning("Skipping API connection test because GOOGLE_API_KEY is not set.")
        logger.info("--- Import Test Successful (API connection not tested) ---")
        return True

if __name__ == "__main__":
    success = run_test()
    if success:
        sys.exit(0)
    else:
        sys.exit(1)
