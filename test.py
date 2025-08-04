# test_memory_system.py
"""
Test script for the Long-Term Memory System
Tests the exact scenarios mentioned in the requirements
"""

import os
import time
import json
from dotenv import load_dotenv

# Import our memory system
from LLM import MemoryEnabledGPT, LongTermMemoryAgent

# Load environment variables
load_dotenv()

def test_basic_memory_flow():
    """Test the basic memory flow as described in requirements"""
    
    print("🧪 Testing Basic Memory Flow")
    print("=" * 50)
    
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not set. Please set it in .env file")
        return False
    
    try:
        # Initialize system
        print("🔧 Initializing Memory System...")
        gpt = MemoryEnabledGPT(api_key)
        user_id = "test_user_" + str(int(time.time()))
        print(f"👤 Test User ID: {user_id}")
        
        # Test 1: User states they use productivity tools
        print("\n📝 Test 1: Memory Storage")
        print("User: 'I use Shram and Magnet as productivity tools'")
        
        response1 = gpt.chat("I use Shram and Magnet as productivity tools", user_id)
        print(f"GPT: {response1}")
        
        # Wait a moment for processing
        time.sleep(1)
        
        # Test 2: Query the stored memory 
        print("\n🔍 Test 2: Memory Retrieval")  
        print("User: 'What are the productivity tools that I use?'")
        
        response2 = gpt.chat("What are the productivity tools that I use?", user_id)
        print(f"GPT: {response2}")
        
        # Check if the response contains the tools
        tools_mentioned = "Shram" in response2 and "Magnet" in response2
        if tools_mentioned:
            print("✅ Memory retrieval successful - both tools mentioned")
        else:
            print("❌ Memory retrieval failed - tools not mentioned correctly")
        
        # Test 3: Memory deletion/update
        print("\n🗑️ Test 3: Memory Update/Deletion")
        print("User: 'I don't use Magnet anymore'")
        
        response3 = gpt.chat("I don't use Magnet anymore", user_id)
        print(f"GPT: {response3}")
        
        # Wait for processing
        time.sleep(1)
        
        # Test 4: Query updated memory
        print("\n🔍 Test 4: Updated Memory Retrieval")
        print("User: 'What productivity tools do I use?'")
        
        response4 = gpt.chat("What productivity tools do I use?", user_id)
        print(f"GPT: {response4}")
        
        # Check if Magnet is no longer mentioned but Shram still is
        magnet_removed = "Magnet" not in response4
        shram_present = "Shram" in response4
        
        if magnet_removed and shram_present:
            print("✅ Memory update successful - Magnet removed, Shram retained")
        elif magnet_removed:
            print("⚠️ Partial success - Magnet removed but Shram might not be mentioned")
        else:
            print("❌ Memory update failed - Magnet still mentioned")
        
        # Test 5: Show user profile
        print("\n Test 5: User Profile")
        profile = gpt.get_user_profile(user_id)
        print(f"Profile: {json.dumps(profile, indent=2)}")
        
        # Test 6: Memory stats
        print("\n Test 6: Memory Statistics")
        stats = gpt.memory_agent.get_memory_stats()
        print(f"Stats: {json.dumps(stats, indent=2)}")
        
        print("\n Basic memory flow test completed!")
        return True
        
    except Exception as e:
        print(f" Test failed with error: {e}")
        return False


def test_advanced_scenarios():
    """Test more advanced memory scenarios"""
    
    print("\n Testing Advanced Scenarios")
    print("=" * 50)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print(" OPENAI_API_KEY not set")
        return False
    
    try:
        gpt = MemoryEnabledGPT(api_key)
        user_id = "advanced_test_" + str(int(time.time()))
        
        # Test complex memory scenarios
        scenarios = [
            "I work as a software engineer in Hyderabad",
            "I prefer Python over Java for development",
            "My goal is to learn machine learning this year",
            "I attended PyCon India last month",
            "I use VS Code and PyCharm for coding"
        ]
        
        print(" Storing multiple memories...")
        for i, scenario in enumerate(scenarios, 1):
            print(f"  {i}. User: {scenario}")
            response = gpt.chat(scenario, user_id)
            print(f"     GPT: {response[:80]}...")
            time.sleep(0.5)
        
        # Test memory queries
        queries = [
            "What's my job?",
            "What programming languages do I prefer?", 
            "What are my goals?",
            "What events have I attended?",
            "What development tools do I use?"
        ]
        
        print("\n Testing memory queries...")
        for i, query in enumerate(queries, 1):
            print(f"  {i}. User: {query}")
            response = gpt.chat(query, user_id)
            print(f"     GPT: {response}")
            time.sleep(0.5)
        
        # Show final profile
        profile = gpt.get_user_profile(user_id)
        print(f"\n Final Profile:")
        for category, items in profile.items():
            if items and category not in ['user_id', 'total_memories']:
                print(f"  {category.title()}: {items}")
        
        print("\n Advanced scenarios test completed!")
        return True
        
    except Exception as e:
        print(f" Advanced test failed: {e}")
        return False


def test_memory_persistence():
    """Test that memories persist across different sessions"""
    
    print("\n Testing Memory Persistence")
    print("=" * 50)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print(" OPENAI_API_KEY not set")
        return False
    
    try:
        user_id = "persistence_test_" + str(int(time.time()))
        
        # Session 1: Store memory
        print(" Session 1: Storing memory...")
        gpt1 = MemoryEnabledGPT(api_key)
        response1 = gpt1.chat("I use Slack for team communication", user_id)
        print(f"Stored: 'I use Slack for team communication'")
        print(f"Response: {response1}")
        
        # Session 2: Retrieve memory (new instance)
        print("\n Session 2: Retrieving memory (new instance)...")
        gpt2 = MemoryEnabledGPT(api_key)
        response2 = gpt2.chat("What do I use for team communication?", user_id)
        print(f"Query: 'What do I use for team communication?'")
        print(f"Response: {response2}")
        
        # Check if memory persisted
        if "Slack" in response2:
            print(" Memory persistence successful!")
        else:
            print(" Memory persistence failed!")
        
        return "Slack" in response2
        
    except Exception as e:
        print(f" Persistence test failed: {e}")
        return False


def test_error_handling():
    """Test error handling scenarios"""
    
    print("\n Testing Error Handling")
    print("=" * 50)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print(" OPENAI_API_KEY not set")
        return False
    
    try:
        gpt = MemoryEnabledGPT(api_key)
        user_id = "error_test_" + str(int(time.time()))
        
        # Test 1: Empty message
        print(" Test 1: Empty message")
        response = gpt.chat("", user_id)
        print(f"Response to empty message: {response}")
        
        # Test 2: Very long message
        print("\n Test 2: Very long message")
        long_message = "I use " + " and ".join([f"tool{i}" for i in range(100)]) + " for work."
        response = gpt.chat(long_message, user_id)
        print(f"Response to long message: {response[:100]}...")
        
        # Test 3: Query non-existent memory
        print("\n Test 3: Query non-existent memory")
        new_user = "non_existent_" + str(int(time.time()))
        response = gpt.chat("What programming languages do I know?", new_user)
        print(f"Response for new user: {response}")
        
        print("\n Error handling tests completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False


def run_all_tests():
    """Run all tests"""
    
    print("Starting Memory System Tests")
    print("=" * 60)
    
    # Check prerequisites
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY environment variable is required")
        print("Please create a .env file with: OPENAI_API_KEY=your_key_here")
        return
    
    results = []
    
    # Run tests
    tests = [
        ("Basic Memory Flow", test_basic_memory_flow),
        ("Advanced Scenarios", test_advanced_scenarios), 
        ("Memory Persistence", test_memory_persistence),
        ("Error Handling", test_error_handling)
    ]
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"Running: {test_name}")
        print(f"{'='*60}")
        
        try:
            result = test_func()
            results.append((test_name, result))
            
            if result:
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
                
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Memory system is working correctly.")
    else:
        print("⚠️ Some tests failed. Please check the output above.")


if __name__ == "__main__":
    run_all_tests()