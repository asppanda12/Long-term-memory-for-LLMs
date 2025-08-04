# api_server.py
"""
Flask API for Memory-Enabled GPT System
Provides REST endpoints for chat with long-term memory
"""

import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import logging

from LLM import MemoryEnabledGPT, LongTermMemoryAgent
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Initialize memory-enabled GPT
try:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    
    memory_gpt = MemoryEnabledGPT(api_key)
    logger.info("Memory-enabled GPT initialized successfully")
    
except Exception as e:
    logger.error(f"Failed to initialize GPT: {e}")
    memory_gpt = None


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    if memory_gpt is None:
        return jsonify({"status": "unhealthy", "error": "GPT not initialized"}), 500
    
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "memory_stats": memory_gpt.memory_agent.get_memory_stats()
    })


@app.route('/chat', methods=['POST'])
def chat():
    """
    Chat with memory-enabled GPT
    
    Request body:
    {
        "message": "user message",
        "user_id": "unique_user_identifier",
        "use_memory": true
    }
    """
    if memory_gpt is None:
        return jsonify({"error": "Service unavailable"}), 503
    
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({"error": "Message is required"}), 400
        
        message = data['message']
        user_id = data.get('user_id', 'default_user')
        use_memory = data.get('use_memory', True)
        
        if not message.strip():
            return jsonify({"error": "Message cannot be empty"}), 400
        
        # Get response from memory-enabled GPT
        response = memory_gpt.chat(message, user_id, use_memory)
        
        return jsonify({
            "response": response,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "memory_enabled": use_memory
        })
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route('/user/<user_id>/profile', methods=['GET'])
def get_user_profile(user_id):
    """Get comprehensive user profile with all memories"""
    if memory_gpt is None:
        return jsonify({"error": "Service unavailable"}), 503
    
    try:
        profile = memory_gpt.get_user_profile(user_id)
        return jsonify(profile)
        
    except Exception as e:
        logger.error(f"Error getting user profile: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route('/user/<user_id>/memories', methods=['GET'])
def get_user_memories(user_id):
    """Get all memories for a specific user"""
    if memory_gpt is None:
        return jsonify({"error": "Service unavailable"}), 503
    
    try:
        memory_type = request.args.get('type')  # Optional filter by memory type
        
        if memory_type:
            from llm_memory_system_openai import MemoryType
            try:
                filter_type = MemoryType(memory_type)
                memories = memory_gpt.memory_agent.get_user_memories(user_id, filter_type)
            except ValueError:
                return jsonify({"error": f"Invalid memory type: {memory_type}"}), 400
        else:
            memories = memory_gpt.memory_agent.get_user_memories(user_id)
        
        return jsonify({
            "user_id": user_id,
            "total_memories": len(memories),
            "memories": memories,
            "filtered_by_type": memory_type
        })
        
    except Exception as e:
        logger.error(f"Error getting user memories: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route('/user/<user_id>/memories', methods=['DELETE'])
def delete_user_memories(user_id):
    """Delete all memories for a specific user"""
    if memory_gpt is None:
        return jsonify({"error": "Service unavailable"}), 503
    
    try:
        success = memory_gpt.clear_user_memory(user_id)
        
        if success:
            return jsonify({
                "message": f"All memories for user {user_id} deleted successfully",
                "user_id": user_id
            })
        else:
            return jsonify({"error": "No memories found to delete"}), 404
            
    except Exception as e:
        logger.error(f"Error deleting user memories: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route('/search', methods=['POST'])
def search_memories():
    """
    Search memories for a user
    
    Request body:
    {
        "query": "search query",
        "user_id": "user_identifier",
        "top_k": 5,
        "threshold": 0.7
    }
    """
    if memory_gpt is None:
        return jsonify({"error": "Service unavailable"}), 503
    
    try:
        data = request.get_json()
        
        if not data or 'query' not in data or 'user_id' not in data:
            return jsonify({"error": "Query and user_id are required"}), 400
        
        query = data['query']
        user_id = data['user_id']
        top_k = data.get('top_k', 5)
        threshold = data.get('threshold', 0.7)
        
        # Search memories
        results = memory_gpt.memory_agent.search_similar_memories(query, user_id, top_k)
        
        # Filter by threshold
        filtered_results = [r for r in results if r['similarity'] >= threshold]
        
        return jsonify({
            "query": query,
            "user_id": user_id,
            "total_found": len(filtered_results),
            "results": filtered_results,
            "threshold": threshold
        })
        
    except Exception as e:
        logger.error(f"Error searching memories: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route('/ask', methods=['POST'])
def ask_memory():
    """
    Ask a question using only stored memories (no GPT conversation)
    
    Request body:
    {
        "question": "What tools do I use?",
        "user_id": "user_identifier"
    }
    """
    if memory_gpt is None:
        return jsonify({"error": "Service unavailable"}), 503
    
    try:
        data = request.get_json()
        
        if not data or 'question' not in data or 'user_id' not in data:
            return jsonify({"error": "Question and user_id are required"}), 400
        
        question = data['question']
        user_id = data['user_id']
        
        # Get answer from memory
        answer = memory_gpt.memory_agent.answer_with_memory(question, user_id)
        
        # Get the memories that were used
        relevant_memories = memory_gpt.memory_agent.get_relevant_memories(question, user_id)
        
        return jsonify({
            "question": question,
            "answer": answer,
            "user_id": user_id,
            "memories_used": len(relevant_memories),
            "relevant_memories": relevant_memories[:3],  # Show top 3 memories used
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error answering from memory: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route('/memories/add', methods=['POST'])
def add_memory():
    """
    Manually add a memory
    
    Request body:
    {
        "content": "User uses VS Code for development",
        "user_id": "user_identifier",
        "memory_type": "tool_usage",
        "confidence": 0.9,
        "metadata": {"category": "development"}
    }
    """
    if memory_gpt is None:
        return jsonify({"error": "Service unavailable"}), 503
    
    try:
        data = request.get_json()
        
        if not data or 'content' not in data or 'user_id' not in data:
            return jsonify({"error": "Content and user_id are required"}), 400
        
        from llm_memory_system_openai import Memory, MemoryType
        import uuid
        
        # Create memory object
        memory = Memory(
            id=str(uuid.uuid4()),
            user_id=data['user_id'],
            content=data['content'],
            memory_type=MemoryType(data.get('memory_type', 'fact')),
            confidence=data.get('confidence', 0.8),
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            metadata={
                **data.get('metadata', {}),
                'action': 'add',
                'source': 'manual'
            }
        )
        
        # Store memory
        success = memory_gpt.memory_agent.store_memory(memory)
        
        if success:
            return jsonify({
                "message": "Memory added successfully",
                "memory_id": memory.id,
                "content": memory.content
            })
        else:
            return jsonify({"error": "Failed to store memory"}), 500
            
    except ValueError as e:
        return jsonify({"error": f"Invalid memory type: {e}"}), 400
    except Exception as e:
        logger.error(f"Error adding memory: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route('/stats', methods=['GET'])
def get_system_stats():
    """Get overall system statistics"""
    if memory_gpt is None:
        return jsonify({"error": "Service unavailable"}), 503
    
    try:
        stats = memory_gpt.memory_agent.get_memory_stats()
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route('/demo', methods=['GET'])
def run_demo():
    """Run a demo conversation to test the system"""
    if memory_gpt is None:
        return jsonify({"error": "Service unavailable"}), 503
    
    try:
        demo_user = "demo_user_" + str(int(datetime.now().timestamp()))
        
        # Demo conversation steps
        steps = [
            "I use Shram and Magnet as my productivity tools",
            "I work as a software engineer in Hyderabad", 
            "What productivity tools do I use?",
            "Actually, I don't use Magnet anymore. I switched to Notion.",
            "What productivity tools do I use now?"
        ]
        
        results = []
        
        for step in steps:
            response = memory_gpt.chat(step, demo_user)
            results.append({
                "user_message": step,
                "gpt_response": response
            })
        
        # Get final profile
        profile = memory_gpt.get_user_profile(demo_user)
        
        return jsonify({
            "demo_user_id": demo_user,
            "conversation_steps": results,
            "final_profile": profile,
            "message": "Demo completed successfully"
        })
        
    except Exception as e:
        logger.error(f"Error running demo: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == '__main__':
    # Ensure OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        exit(1)
    
    # Get port from environment or default to 5000
    port = int(os.getenv('PORT', 5000))
    debug_mode = os.getenv('FLASK_ENV') == 'development'
    
    print(f"Starting Memory-Enabled GPT API on port {port}")
    print(f"Memory database path: ./memory_db")
    print(f"OpenAI API key: {'✓ Set' if os.getenv('OPENAI_API_KEY') else '✗ Missing'}")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug_mode
    )