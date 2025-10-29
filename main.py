#!/usr/bin/env python3
"""
Test Event Similarity Analysis - Standalone Test Runner

This script tests the event similarity analysis functionality without FastAPI.
Configure the variables below and run to test the complete workflow.
"""

import sys
import os
import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =================== TEST CONFIGURATION ===================
# Modify these variables for your test case

# Event Information
EVENT_NAME = "Happy Hundred "
GAME_CODE = "Rise of Kingdoms"
EVENT_ABOUT = """
# About Happy Hundred **A hundred versions, countless epic tales. This journey's only just begun.** ## Overview Log in each day to receive valuable rewards! The Happy Hundred event celebrates milestone achievements with generous daily login bonuses spanning multiple weeks of gameplay. ## Event Duration - **Start Date:** October 11, 2025 (UTC) - **End Date:** November 9, 2025 (UTC) - **Total Duration:** 29 days ## Requirements Your City Hall must be at least level 8 to participate in this event. ## How to Play 1. During this event, you can claim a reward each day you log in to the game. 2. The event features a 28-day reward calendar with increasingly valuable prizes as you progress. 3. If you forget to log in on a certain date, you can spend Gems to make up for the missed day. Note that the more days you've made up, the higher the cost will be. ## Reward Structure The event offers daily rewards including: - **Resources:** Large amounts of Food, Wood, and other essential materials - **Speed-ups:** Various time acceleration items to boost your city's development - **Special Items:** Exclusive rewards available only during this celebration Key milestone rewards are available on Days 3, 7, 14, 21, and 28, offering particularly valuable prizes for consistent participation. Don't miss this opportunity to strengthen your kingdom with substantial daily rewards while celebrating Rise of Kingdoms' milestone achievements!


"""

# Image Folder Configuration
FOLDER_NAME = "happy_hundred_rise_of_kingdoms"  # Folder containing test images
SHARED_UPLOADS_PATH = "/media/luanpc/Video/shared/uploads/"  # Base path for uploads
EXPECTED_IMAGE_COUNT =  2 # Expected number of images in folder

# Test Options
VERBOSE_OUTPUT = True  # Print detailed results
SAVE_RESULTS = True   # Save results to JSON file
OUTPUT_FILE = "test_results.json"  # Output file name

# =================== INITIALIZATION FUNCTIONS ===================

async def initialize_services():
    """Initialize all required services for testing"""
    print("🔧 Initializing services...")
    
    services = {}
    
    try:
        # Initialize Places365 model
        print("Loading Places365 model...")
        from models.places365 import get_places365_model
        places_model = get_places365_model()
        services['places365'] = places_model is not None
        print(f"✅ Places365 model: {'loaded' if services['places365'] else 'failed'}")
        
        # Initialize Voyage client
        print("Initializing Voyage client...")
        from utils.text_processor import get_voyage_client
        voyage_client = get_voyage_client()
        services['voyage'] = voyage_client is not None
        services['voyage_client'] = voyage_client
        print(f"Voyage client: {'initialized' if services['voyage'] else 'failed'}")
        
        # Initialize Claude service
        print("Testing Claude service...")
        try:
            from services.claude_service import ClaudeService
            # Create Claude service instance properly
            claude_service = ClaudeService()
            claude_status = claude_service.get_usage_stats()
            services['claude'] = True
            services['claude_service'] = claude_service
            print(f"✅ Claude service: ready ({claude_status.get('model', 'unknown model')})")
        except Exception as e:
            services['claude'] = False
            services['claude_service'] = None
            print(f"⚠️ Claude service: {e}")
            if VERBOSE_OUTPUT:
                print(f"   Make sure CLAUDE_API_KEY is set in .env file")
        
        # Initialize Database service (THIS WAS MISSING!)
        print("Initializing Database service...")
        try:
            from services.database_service import DatabaseService
            db_service = DatabaseService()
            await db_service.initialize()
            
            # Test database connection
            db_health = await db_service.health_check()
            services['database'] = db_health.get("status") == "healthy"
            services['database_service'] = db_service
            print(f"✅ Database service: {'connected' if services['database'] else 'failed'}")
        except Exception as e:
            services['database'] = False
            services['database_service'] = None
            print(f"⚠️ Database service: {e}")
        
        # Initialize Event Similarity service (MOVED TO AFTER DATABASE INIT)
        print("Initializing Event Similarity service...")
        try:
            # Check if config files exist first
            config_path = Path("config")
            required_configs = [
                "prompts.yaml", 
                "output_formats.yaml",
                "similarity_prompts.yaml", 
                "similarity_output_formats.yaml"
            ]
            
            missing_configs = []
            for config_file in required_configs:
                if not (config_path / config_file).exists():
                    missing_configs.append(config_file)
            
            if missing_configs:
                print(f"⚠️ Missing config files: {missing_configs}")
                print(f"   Expected location: {config_path.absolute()}/")
            
            # Import required classes
            from services.event_similarity_service import EventSimilarityService
            from utils.prompt_manager import PromptManager
            
            # Check that all dependencies are available
            if not services.get('claude_service'):
                raise Exception("Claude service not available")
            if not services.get('voyage_client'):
                raise Exception("Voyage client not available") 
            if not services.get('database_service'):
                raise Exception("Database service not available")
            
            # Initialize PromptManager
            prompt_manager = PromptManager()
            
            # Create service instance with ALL required dependencies
            event_similarity_service = EventSimilarityService(
                claude_service=services['claude_service'],      # Fixed: pass claude_service
                voyage_client=services['voyage_client'],        # Fixed: pass voyage_client  
                prompt_manager=prompt_manager,                  # Fixed: initialize and pass prompt_manager
                database_service=services['database_service']   # Fixed: ensure database_service exists
            )
            
            # Test service status
            service_status = await event_similarity_service.get_service_status()
            services['event_similarity'] = service_status.get("status") == "operational"
            services['event_similarity_service'] = event_similarity_service
            print(f"✅ Event Similarity service: {'ready' if services['event_similarity'] else 'failed'}")
            
            if VERBOSE_OUTPUT:
                db_status = service_status.get('database_health', {})
                print(f"   Database connected: {db_status.get('status') == 'healthy'}")
                faiss_stats = service_status.get('faiss_stats', {})
                faiss_count = len([k for k, v in faiss_stats.items() if isinstance(v, dict) and v.get('total_vectors', 0) > 0])
                print(f"   FAISS indexes loaded: {faiss_count}")
                
        except ImportError as e:
            services['event_similarity'] = False
            services['event_similarity_service'] = None
            print(f"Event Similarity service - Import Error: {e}")
            if "sklearn" in str(e):
                print(f"   Solution: pip install scikit-learn==1.3.2")
            elif "asyncpg" in str(e):
                print(f"   Solution: pip install asyncpg==0.29.0")
            elif "Dict" in str(e):
                print(f"   Solution: Add 'from typing import Dict' to utils/prompt_manager.py")
                print(f"   File location: utils/prompt_manager.py")
        except Exception as e:
            services['event_similarity'] = False
            services['event_similarity_service'] = None
            if "ufunc" in str(e) or "scipy" in str(e):
                print(f"Event Similarity service - scipy/numpy compatibility issue")
                print(f"   Solution: pip install scipy==1.10.1 numpy==1.24.3")
                print(f"   Or try: pip uninstall numpy scipy -y && pip install numpy==1.24.3 scipy==1.10.1")
            else:
                print(f"Event Similarity service error: {e}")
                if VERBOSE_OUTPUT:
                    logger.exception("Detailed Event Similarity service error:")
        
        return services
        
    except Exception as e:
        print(f"❌ Service initialization failed: {e}")
        return {}

def validate_test_configuration():
    """Validate test configuration and folder existence"""
    print("🔍 Validating test configuration...")
    
    # Check required environment variables
    required_env_vars = ['CLAUDE_API_KEY', 'VOYAGE_API_KEY', 'DB_HOST', 'DB_USER', 'DB_PASS', 'DB_NAME']
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing environment variables: {missing_vars}")
        return False
    
    # Check image folder
    folder_path = Path(SHARED_UPLOADS_PATH) / FOLDER_NAME
    if not folder_path.exists():
        print(f"❌ Image folder not found: {folder_path}")
        print(f"   Please create the folder and add {EXPECTED_IMAGE_COUNT} test images")
        return False
    
    # Check images in folder
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = [
        f for f in folder_path.iterdir() 
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    
    if len(image_files) == 0:
        print(f"❌ No image files found in folder: {folder_path}")
        return False
    
    if len(image_files) != EXPECTED_IMAGE_COUNT:
        print(f"⚠️ Image count mismatch: expected {EXPECTED_IMAGE_COUNT}, found {len(image_files)}")
        print(f"   Found images: {[f.name for f in image_files]}")
        print(f"   Continuing with actual count: {len(image_files)}")
        # We'll use the actual count in the test instead of global modification
    
    print(f"✅ Configuration valid")
    print(f"   Event: {EVENT_NAME}")
    print(f"   Game: {GAME_CODE}")
    print(f"   Images: {EXPECTED_IMAGE_COUNT} files in {folder_path}")
    
    return len(image_files)  # Return actual image count

# =================== TEST EXECUTION ===================

async def run_event_similarity_test(services, actual_image_count):
    """Run the complete event similarity analysis test"""
    print("\n🚀 Starting Event Similarity Analysis Test...")
    print("=" * 60)
    
    if not services.get('event_similarity_service'):
        print("❌ Event Similarity service not available")
        return None
    
    # Prepare request parameters
    test_request = {
        "folder_name": FOLDER_NAME,
        "game_code": GAME_CODE,
        "event_name": EVENT_NAME,
        "about": EVENT_ABOUT,
        "image_count": actual_image_count,  # Use actual count from validation
        "shared_uploads_path": SHARED_UPLOADS_PATH
    }
    
    print("📋 Test Parameters:")
    print(f"   Folder: {test_request['folder_name']}")
    print(f"   Game: {test_request['game_code']}")
    print(f"   Event: {test_request['event_name']}")
    print(f"   Images: {test_request['image_count']}")
    print(f"   Path: {test_request['shared_uploads_path']}")
    print()
    
    try:
        # Run the analysis
        print("🔄 Running similarity analysis...")
        start_time = asyncio.get_event_loop().time()
        
        result = await services['event_similarity_service'].find_similar_events(
                query_name=test_request['event_name'],        
                query_about=test_request['about'],            
                folder_name=test_request['folder_name'],
                game_code=test_request['game_code'],
                shared_uploads_path=test_request['shared_uploads_path'],
                image_count=test_request['image_count']
        )
        
        end_time = asyncio.get_event_loop().time()
        execution_time = end_time - start_time
        
        print(f"✅ Analysis completed in {execution_time:.2f} seconds")
        
        # Display results
        display_results(result, execution_time)
        
        # Save results if requested
        if SAVE_RESULTS:
            save_test_results(test_request, result, execution_time)
        
        return result
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        logger.exception("Detailed error information:")
        return None

def display_results(result, execution_time):
    """Display test results in a formatted way"""
    print("\n📊 TEST RESULTS")
    print("=" * 60)
    
    # Query event information
    query_event = result.get('query_event', {})
    print("🎯 QUERY EVENT:")
    print(f"   Name: {query_event.get('name', 'N/A')}")
    print(f"   About: {query_event.get('about', 'N/A')[:100]}...")
    
    tags = query_event.get('tags', {})
    print(f"   Tags:")
    print(f"     Family: {tags.get('family', 'N/A')}")
    print(f"     Dynamics: {tags.get('dynamics', 'N/A')}")
    print(f"     Rewards: {tags.get('rewards', 'N/A')}")
    
    if VERBOSE_OUTPUT:
        print(f"   Tag Explanation: {query_event.get('tag_explanation', 'N/A')}")
    
    # Similar events
    similar_events = result.get('similar_events', [])
    print(f"\n🔍 SIMILAR EVENTS FOUND: {len(similar_events)}")
    
    if similar_events:
        print("\nTop Similar Events:")
        for i, event in enumerate(similar_events[:5], 1):  # Show top 5
            print(f"\n{i}. {event.get('name', 'N/A')}")
            print(f"   Text Score: {event.get('score_text', 0.0):.3f}")
            print(f"   Image Score: {event.get('score_image', 0.0):.3f}")
            print(f"   Matching Images: {len(event.get('image_faiss_indices', []))} indices")
            
            event_tags = event.get('tags', {})
            print(f"   Tags: {event_tags.get('family', 'N/A')} | {event_tags.get('dynamics', 'N/A')} | {event_tags.get('rewards', 'N/A')}")
            
            if VERBOSE_OUTPUT:
                print(f"   Reason: {event.get('reason', 'N/A')[:150]}...")
                print(f"   About: {event.get('about', 'N/A')[:100]}...")
        
        # Statistics
        text_scores = [e.get('score_text', 0.0) for e in similar_events]
        image_scores = [e.get('score_image', 0.0) for e in similar_events]
        
        print(f"\n📈 STATISTICS:")
        print(f"   Total Events Found: {len(similar_events)}")
        print(f"   Average Text Score: {sum(text_scores)/len(text_scores):.3f}")
        print(f"   Average Image Score: {sum(image_scores)/len(image_scores):.3f}")
        print(f"   Max Text Score: {max(text_scores):.3f}")
        print(f"   Max Image Score: {max(image_scores):.3f}")
        print(f"   Execution Time: {execution_time:.2f} seconds")
    else:
        print("   No similar events found")

def save_test_results(request, result, execution_time):
    """Save test results to JSON file"""
    try:
        output_data = {
            "test_configuration": {
                "timestamp": asyncio.get_event_loop().time(),
                "execution_time_seconds": execution_time,
                "request_parameters": request,
                "config": {
                    "event_name": EVENT_NAME,
                    "game_code": GAME_CODE,
                    "folder_name": FOLDER_NAME,
                    "expected_image_count": EXPECTED_IMAGE_COUNT
                }
            },
            "results": result
        }
        
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {OUTPUT_FILE}")
        
    except Exception as e:
        print(f"⚠️ Failed to save results: {e}")

# =================== MAIN TEST RUNNER ===================

async def main():
    """Main test runner"""
    print("🧪 EVENT SIMILARITY ANALYSIS TEST")
    print("=" * 60)
    print(f"Event: {EVENT_NAME}")
    print(f"Game: {GAME_CODE}")  
    print(f"Folder: {FOLDER_NAME}")
    print("=" * 60)
    
    # Validate configuration
    actual_image_count = validate_test_configuration()
    if not actual_image_count:
        print("❌ Configuration validation failed. Please fix the issues above.")
        return
    
    # Initialize services
    services = await initialize_services()
    
    # Check if critical services are ready
    critical_services = ['places365', 'voyage', 'claude', 'database', 'event_similarity']
    missing_services = [s for s in critical_services if not services.get(s, False)]
    
    if missing_services:
        print(f"\n⚠️ Warning: Some services are not available: {missing_services}")
        print("The test may fail or produce incomplete results.")
        
        if 'event_similarity' in missing_services:
            print("❌ Event Similarity service is required. Cannot proceed.")
            return
        
        input("Press Enter to continue anyway, or Ctrl+C to abort...")
    
    print(f"\n✅ All services initialized successfully")
    
    # Run the test
    try:
        result = await run_event_similarity_test(services, actual_image_count)
        
        if result:
            print("\n🎉 Test completed successfully!")
        else:
            print("\n❌ Test failed!")
            
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error during test: {e}")
        logger.exception("Detailed error:")
    finally:
        # Cleanup
        if services.get('database_service'):
            try:
                await services['database_service'].close()
                print("🧹 Database connections closed")
            except:
                pass

if __name__ == "__main__":
    # Clear screen for better readability
    os.system('clear' if os.name == 'posix' else 'cls')
    
    # Run the test
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Test aborted by user")
    except Exception as e:
        print(f"\n❌ Critical error: {e}")
    
    print("\n" + "=" * 60)
    print("Test completed. Check the output above for results.")