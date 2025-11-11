#!/usr/bin/env python3
"""
Event Similarity Analysis - Standalone Test Runner

Refactored version using modular architecture.
Configure the test in tests/test_config.py and run to test the complete workflow.
"""

import sys
import os
import asyncio
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import configuration
from config import config

# Setup logging using config
config.setup_logging()
logger = logging.getLogger(__name__)

# Import test modules
from tests.test_config import TestConfig, validate_test_configuration, get_test_request
from tests.service_initializer import initialize_services, validate_critical_services
from tests.test_runner import run_event_similarity_test
from tests.test_reporter import display_results, save_test_results


async def main():
    """Main test runner"""
    print("🧪 EVENT SIMILARITY ANALYSIS TEST")
    print("=" * 60)

    # Create test configuration
    test_config = TestConfig()

    print(f"Event: {test_config.EVENT_NAME}")
    print(f"Game: {test_config.GAME_CODE}")
    print(f"Folder: {test_config.FOLDER_NAME}")
    print("=" * 60)

    # Validate configuration
    actual_image_count = validate_test_configuration(test_config)
    if not actual_image_count:
        print("❌ Configuration validation failed. Please fix the issues above.")
        return

    # Initialize services
    services = await initialize_services(verbose=test_config.VERBOSE_OUTPUT)

    # Check if critical services are ready
    if not validate_critical_services(services, verbose=test_config.VERBOSE_OUTPUT):
        return

    # Prepare test request
    test_request = get_test_request(test_config, actual_image_count)

    # Run the test
    try:
        test_output = await run_event_similarity_test(
            services,
            test_request,
            verbose=test_config.VERBOSE_OUTPUT
        )

        if test_output:
            result = test_output['result']
            execution_time = test_output['execution_time']

            # Display results
            display_results(result, execution_time, verbose=test_config.VERBOSE_OUTPUT)

            # Save results if requested
            if test_config.SAVE_RESULTS:
                save_test_results(
                    test_request,
                    result,
                    execution_time,
                    output_file=test_config.OUTPUT_FILE,
                    config=test_config
                )

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

    # Print current configuration
    if config.IS_DEVELOPMENT:
        config.print_config()

    # Run the test
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Test aborted by user")
    except Exception as e:
        print(f"\n❌ Critical error: {e}")
        logger.exception("Critical error:")

    print("\n" + "=" * 60)
    print("Test completed. Check the output above for results.")
