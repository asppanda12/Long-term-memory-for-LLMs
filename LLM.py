# requirements.txt
"""
openai>=1.0.0
chromadb>=0.4.0
python-dotenv>=1.0.0
tiktoken>=0.5.0
pydantic>=2.0.0
numpy>=1.21.0
"""

import os
import json
import uuid
import time
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging

import openai
import chromadb
from chromadb.config import Settings
import tiktoken
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MemoryType(Enum):
    PREFERENCE = "preference"
    FACT = "fact" 
    TOOL_USAGE = "tool_usage"
    GOAL = "goal"
    EXPERIENCE = "experience"


@dataclass
class Memory:
    id: str
    user_id: str
    content: str
    memory_type: MemoryType
    confidence: float
    created_at: str
    updated_at: str
    metadata: Dict[str, Any]


class LongTermMemoryAgent:
    """
    Long-term memory system that integrates with OpenAI APIs
    to provide persistent memory for GPT conversations.
    """
    
    def __init__(
        self,
        openai_api_key: str,
        db_path: str = "./memory_db",
        collection_name: str = "user_memories",
        embedding_model: str = "text-embedding-3-small",
        chat_model: str = "gpt-3.5-turbo"
    ):
        """
        Initialize the memory agent
        
        Args:
            openai_api_key: OpenAI API key
            db_path: Path to store ChromaDB
            collection_name: Name for the memory collection
            embedding_model: OpenAI embedding model
            chat_model: OpenAI chat model for memory extraction
        """
        # Set up OpenAI
        openai.api_key = openai_api_key
        self.client = openai.OpenAI(api_key=openai_api_key)
        self.embedding_model = embedding_model
        self.chat_model = chat_model
        
        # Set up ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        try:
            self.collection = self.chroma_client.get_collection(collection_name)
        except:
            self.collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        
        # Token encoder for efficiency
        self.encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
        
        logger.info(f"Memory agent initialized with {self.collection.count()} existing memories")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoder.encode(text))
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using OpenAI"""
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return []
    
    def extract_memories(self, conversation: List[Dict[str, str]], user_id: str) -> List[Memory]:
        """
        Extract memories from conversation using OpenAI
        
        Args:
            conversation: List of messages [{"role": "user/assistant", "content": "..."}]
            user_id: User identifier
            
        Returns:
            List of extracted Memory objects
        """
        # Convert conversation to text
        conv_text = self._conversation_to_text(conversation)
        
        # Limit tokens to avoid API limits
        if self.count_tokens(conv_text) > 3000:
            conv_text = conv_text[-3000:]
        
        system_prompt = """You are a memory extraction agent. Extract important, factual memories about the user from this conversation.

MEMORY TYPES:
- preference: User likes/dislikes, choices (e.g., "prefers dark mode")
- fact: Personal facts about user (e.g., "works as engineer", "lives in NYC")  
- tool_usage: Tools, apps, software user mentions using (e.g., "uses VS Code")
- goal: User objectives, plans (e.g., "wants to learn Python")
- experience: Past experiences, events (e.g., "attended conference")

RULES:
1. Only extract information the USER states about THEMSELVES
2. Be specific and factual
3. Ignore temporary states ("I'm tired today")
4. Focus on lasting, useful information

For each memory, determine if it should be:
- ADD: New memory to store
- UPDATE: Modify existing memory 
- DELETE: Remove memory (when user says "don't use X anymore")

Return JSON array:
[
  {
    "content": "User uses Shram and Magnet as productivity tools",
    "memory_type": "tool_usage", 
    "action": "add",
    "confidence": 0.95,
    "metadata": {"tools": ["Shram", "Magnet"], "category": "productivity"}
  }
]

If no memories found, return empty array []."""

        try:
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Conversation:\n{conv_text}"}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            # Parse response
            content = response.choices[0].message.content.strip()
            
            # Handle cases where response might have markdown formatting
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "").strip()
            
            memory_data = json.loads(content)
            
            memories = []
            for mem_dict in memory_data:
                memory = Memory(
                    id=str(uuid.uuid4()),
                    user_id=user_id,
                    content=mem_dict["content"],
                    memory_type=MemoryType(mem_dict["memory_type"]),
                    confidence=mem_dict.get("confidence", 0.8),
                    created_at=datetime.now().isoformat(),
                    updated_at=datetime.now().isoformat(),
                    metadata={
                        **mem_dict.get("metadata", {}),
                        "action": mem_dict.get("action", "add")
                    }
                )
                memories.append(memory)
            
            logger.info(f"Extracted {len(memories)} memories for user {user_id}")
            return memories
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse memory extraction response: {e}")
            return []
        except Exception as e:
            logger.error(f"Error extracting memories: {e}")
            return []
    
    def _conversation_to_text(self, conversation: List[Dict[str, str]]) -> str:
        """Convert conversation to text format"""
        text_parts = []
        for msg in conversation:
            role = msg["role"].title()
            content = msg["content"]
            text_parts.append(f"{role}: {content}")
        return "\n".join(text_parts)
    
    def store_memory(self, memory: Memory) -> bool:
        """
        Store a memory in the vector database
        
        Args:
            memory: Memory object to store
            
        Returns:
            Success status
        """
        try:
            action = memory.metadata.get("action", "add")
            
            if action == "delete":
                return self._handle_memory_deletion(memory)
            elif action == "update":
                return self._handle_memory_update(memory)
            else:
                return self._add_new_memory(memory)
                
        except Exception as e:
            logger.error(f"Error storing memory: {e}")
            return False
    
    def _add_new_memory(self, memory: Memory) -> bool:
        """Add new memory to database"""
        try:
            # Check for similar existing memories
            similar_memories = self.search_similar_memories(memory.content, memory.user_id, top_k=3)
            
            # If very similar memory exists, update instead of add
            for sim_memory in similar_memories:
                if sim_memory["similarity"] > 0.9:  # Very similar
                    logger.info(f"Updating similar memory instead of adding new one")
                    return self._update_existing_memory(sim_memory["id"], memory)
            
            # Get embedding
            embedding = self.get_embedding(memory.content)
            if not embedding:
                return False
            
            # Store in ChromaDB
            self.collection.add(
                embeddings=[embedding],
                documents=[memory.content],
                metadatas=[{
                    "user_id": memory.user_id,
                    "memory_type": memory.memory_type.value,
                    "confidence": memory.confidence,
                    "created_at": memory.created_at,
                    "updated_at": memory.updated_at,
                    **memory.metadata
                }],
                ids=[memory.id]
            )
            
            logger.info(f"Added memory: {memory.content[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"Error adding memory: {e}")
            return False
    
    def _handle_memory_deletion(self, memory: Memory) -> bool:
        """Handle memory deletion requests"""
        try:
            # Find memories to delete based on content similarity
            similar_memories = self.search_similar_memories(memory.content, memory.user_id, top_k=5)
            
            deleted_count = 0
            for sim_memory in similar_memories:
                if sim_memory["similarity"] > 0.7:  # Reasonably similar
                    self.collection.delete(ids=[sim_memory["id"]])
                    deleted_count += 1
                    logger.info(f"Deleted memory: {sim_memory['content'][:50]}...")
            
            logger.info(f"Deleted {deleted_count} memories based on deletion request")
            return deleted_count > 0
            
        except Exception as e:
            logger.error(f"Error deleting memories: {e}")
            return False
    
    def _handle_memory_update(self, memory: Memory) -> bool:
        """Handle memory update requests"""
        try:
            # Find most similar memory to update
            similar_memories = self.search_similar_memories(memory.content, memory.user_id, top_k=1)
            
            if similar_memories and similar_memories[0]["similarity"] > 0.7:
                # Update existing memory
                return self._update_existing_memory(similar_memories[0]["id"], memory)
            else:
                # No similar memory found, add as new
                return self._add_new_memory(memory)
                
        except Exception as e:
            logger.error(f"Error updating memory: {e}")
            return False
    
    def _update_existing_memory(self, memory_id: str, new_memory: Memory) -> bool:
        """Update an existing memory"""
        try:
            # Delete old memory
            self.collection.delete(ids=[memory_id])
            
            # Add updated memory with new ID
            new_memory.id = str(uuid.uuid4())
            new_memory.updated_at = datetime.now().isoformat()
            
            return self._add_new_memory(new_memory)
            
        except Exception as e:
            logger.error(f"Error updating existing memory: {e}")
            return False
    
    def search_similar_memories(self, query: str, user_id: str, top_k: int = 5) -> List[Dict]:
        """
        Search for memories similar to query
        
        Args:
            query: Search query
            user_id: User identifier
            top_k: Number of results
            
        Returns:
            List of similar memories with similarity scores
        """
        try:
            # Get query embedding
            query_embedding = self.get_embedding(query)
            if not query_embedding:
                return []
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where={"user_id": user_id},
                include=["documents", "metadatas", "distances"]
            )
            
            memories = []
            if results["documents"] and results["documents"][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results["documents"][0],
                    results["metadatas"][0], 
                    results["distances"][0]
                )):
                    similarity = 1 - distance  # Convert distance to similarity
                    memories.append({
                        "id": results["ids"][0][i],
                        "content": doc,
                        "similarity": similarity,
                        "metadata": metadata
                    })
            
            return memories
            
        except Exception as e:
            logger.error(f"Error searching memories: {e}")
            return []
    
    def get_relevant_memories(self, query: str, user_id: str, threshold: float = 0.7) -> List[Dict]:
        """Get memories relevant to a query above similarity threshold"""
        all_memories = self.search_similar_memories(query, user_id, top_k=10)
        return [mem for mem in all_memories if mem["similarity"] >= threshold]
    
    def answer_with_memory(self, question: str, user_id: str) -> str:
        """
        Answer a question using stored memories
        
        Args:
            question: User's question
            user_id: User identifier
            
        Returns:
            Answer based on memories
        """
        try:
            # Get relevant memories
            relevant_memories = self.get_relevant_memories(question, user_id, threshold=0.6)
            
            if not relevant_memories:
                return "I don't have any information stored about that."
            
            # Format memories for GPT
            memory_context = "\n".join([
                f"- {mem['content']} (confidence: {mem['metadata'].get('confidence', 'unknown')})"
                for mem in relevant_memories[:5]  # Limit to top 5
            ])
            
            system_prompt = f"""You are answering a question using the user's stored memories. 

STORED MEMORIES:
{memory_context}

INSTRUCTIONS:
1. Answer based ONLY on the provided memories
2. Be conversational and natural
3. If memories don't contain the answer, say so clearly
4. Don't make up information not in the memories
5. Combine multiple memories if relevant"""

            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                temperature=0.1,
                max_tokens=300
            )
            
            answer = response.choices[0].message.content.strip()
            logger.info(f"Answered question using {len(relevant_memories)} memories")
            
            return answer
            
        except Exception as e:
            logger.error(f"Error answering with memory: {e}")
            return "I encountered an error while retrieving your information."
    
    def process_conversation(self, conversation: List[Dict[str, str]], user_id: str) -> Tuple[List[Memory], int]:
        """
        Process a conversation: extract and store memories
        
        Args:
            conversation: List of messages
            user_id: User identifier
            
        Returns:
            Tuple of (stored_memories, total_extracted)
        """
        # Extract memories
        extracted_memories = self.extract_memories(conversation, user_id)
        
        # Store each memory
        stored_memories = []
        for memory in extracted_memories:
            if self.store_memory(memory):
                stored_memories.append(memory)
        
        logger.info(f"Processed conversation: {len(stored_memories)}/{len(extracted_memories)} memories stored")
        return stored_memories, len(extracted_memories)
    
    def get_user_memories(self, user_id: str, memory_type: Optional[MemoryType] = None) -> List[Dict]:
        """Get all memories for a user"""
        try:
            where_clause = {"user_id": user_id}
            if memory_type:
                where_clause["memory_type"] = memory_type.value
            
            results = self.collection.get(
                where=where_clause,
                include=["documents", "metadatas"]
            )
            
            memories = []
            if results["documents"]:
                for doc, metadata in zip(results["documents"], results["metadatas"]):
                    memories.append({
                        "content": doc,
                        "metadata": metadata
                    })
            
            return memories
            
        except Exception as e:
            logger.error(f"Error getting user memories: {e}")
            return []
    
    def delete_user_memories(self, user_id: str) -> bool:
        """Delete all memories for a user"""
        try:
            # Get all memory IDs for user
            results = self.collection.get(
                where={"user_id": user_id},
                include=["documents"]
            )
            
            if results["ids"]:
                self.collection.delete(ids=results["ids"])
                logger.info(f"Deleted {len(results['ids'])} memories for user {user_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error deleting user memories: {e}")
            return False
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory database statistics"""
        try:
            total_count = self.collection.count()
            
            # Get all memories to analyze
            all_results = self.collection.get(include=["metadatas"])
            
            stats = {
                "total_memories": total_count,
                "memory_types": {},
                "users": set(),
                "avg_confidence": 0
            }
            
            if all_results["metadatas"]:
                confidences = []
                for metadata in all_results["metadatas"]:
                    # Count memory types
                    mem_type = metadata.get("memory_type", "unknown")
                    stats["memory_types"][mem_type] = stats["memory_types"].get(mem_type, 0) + 1
                    
                    # Collect users
                    stats["users"].add(metadata.get("user_id", "unknown"))
                    
                    # Collect confidences
                    if "confidence" in metadata:
                        confidences.append(metadata["confidence"])
                
                stats["unique_users"] = len(stats["users"])
                stats["users"] = list(stats["users"])  # Convert set to list for JSON serialization
                
                if confidences:
                    stats["avg_confidence"] = sum(confidences) / len(confidences)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            return {"error": str(e)}


class MemoryEnabledGPT:
    """
    GPT wrapper with long-term memory capabilities
    """
    
    def __init__(self, openai_api_key: str, memory_agent: Optional[LongTermMemoryAgent] = None):
        """
        Initialize memory-enabled GPT
        
        Args:
            openai_api_key: OpenAI API key
            memory_agent: Optional existing memory agent
        """
        self.client = openai.OpenAI(api_key=openai_api_key)
        self.memory_agent = memory_agent or LongTermMemoryAgent(openai_api_key)
        self.conversation_history = {}  # Store per-user conversation history
    
    def chat(self, message: str, user_id: str, use_memory: bool = True) -> str:
        """
        Chat with memory-enabled GPT
        
        Args:
            message: User message
            user_id: User identifier
            use_memory: Whether to use memory for this interaction
            
        Returns:
            GPT response
        """
        try:
            # Initialize conversation history for user if needed
            if user_id not in self.conversation_history:
                self.conversation_history[user_id] = []
            
            # Check if user is asking for remembered information
            if use_memory and self._is_memory_query(message):
                memory_response = self.memory_agent.answer_with_memory(message, user_id)
                if "don't have any information" not in memory_response:
                    # Add to conversation history
                    self.conversation_history[user_id].append({"role": "user", "content": message})
                    self.conversation_history[user_id].append({"role": "assistant", "content": memory_response})
                    return memory_response
            
            # Regular GPT conversation
            messages = [{"role": "system", "content": "You are a helpful assistant."}]
            
            # Add recent conversation history (last 10 messages)
            recent_history = self.conversation_history[user_id][-10:] if self.conversation_history[user_id] else []
            messages.extend(recent_history)
            
            # Add current message
            messages.append({"role": "user", "content": message})
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            
            assistant_response = response.choices[0].message.content.strip()
            
            # Add to conversation history
            self.conversation_history[user_id].append({"role": "user", "content": message})
            self.conversation_history[user_id].append({"role": "assistant", "content": assistant_response})
            
            # Extract and store memories from recent conversation
            if use_memory:
                recent_conversation = self.conversation_history[user_id][-4:]  # Last 2 exchanges
                stored_memories, _ = self.memory_agent.process_conversation(recent_conversation, user_id)
                
                if stored_memories:
                    logger.info(f"Stored {len(stored_memories)} new memories for user {user_id}")
            
            return assistant_response
            
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return "I encountered an error. Please try again."
    
    def _is_memory_query(self, message: str) -> bool:
        """Check if message is asking for personal information"""
        memory_keywords = [
            "what do i", "what tools", "what software", "what apps", "what are my",
            "my preferences", "my job", "where do i", "what's my", "tell me about my",
            "remember", "told you", "mentioned", "said before", "what did i say"
        ]
        
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in memory_keywords)
    
    def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive user profile"""
        memories = self.memory_agent.get_user_memories(user_id)
        
        profile = {
            "user_id": user_id,
            "total_memories": len(memories),
            "tools": [],
            "preferences": [], 
            "facts": [],
            "goals": [],
            "experiences": []
        }
        
        for memory in memories:
            mem_type = memory["metadata"].get("memory_type")
            content = memory["content"]
            
            if mem_type == "tool_usage":
                profile["tools"].append(content)
            elif mem_type == "preference":
                profile["preferences"].append(content)
            elif mem_type == "fact":
                profile["facts"].append(content)
            elif mem_type == "goal":
                profile["goals"].append(content)
            elif mem_type == "experience":
                profile["experiences"].append(content)
        
        return profile
    
    def clear_user_memory(self, user_id: str) -> bool:
        """Clear all memories for a user"""
        success = self.memory_agent.delete_user_memories(user_id)
        if success and user_id in self.conversation_history:
            del self.conversation_history[user_id]
        return success


# Example usage and testing
def main():
    """Example usage of the memory system"""
    
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    # Initialize memory-enabled GPT
    gpt = MemoryEnabledGPT(api_key)
    user_id = "demo_user"
    
    print("Memory-Enabled GPT Demo")
    print("=" * 40)
    
    # Test scenario 1: User shares tool information
    print("\n👤 User: I use Shram and Magnet as my productivity tools")
    response = gpt.chat("I use Shram and Magnet as my productivity tools", user_id)
    print(f"GPT: {response}")
    
    # Test scenario 2: User shares job information  
    print("\n👤 User: I work as a software engineer in Hyderabad")
    response = gpt.chat("I work as a software engineer in Hyderabad", user_id)
    print(f"GPT: {response}")
    
    # Test scenario 3: Memory recall
    print("\n👤 User: What productivity tools do I use?")
    response = gpt.chat("What productivity tools do I use?", user_id)
    print(f"GPT: {response}")
    
    # Test scenario 4: Memory update/deletion
    print("\n👤 User: Actually, I don't use Magnet anymore. I switched to Notion.")
    response = gpt.chat("Actually, I don't use Magnet anymore. I switched to Notion.", user_id)
    print(f"GPT: {response}")
    
    # Test scenario 5: Updated memory recall
    print("\n👤 User: What productivity tools do I use now?")
    response = gpt.chat("What productivity tools do I use now?", user_id)
    print(f"GPT: {response}")
    
    # Show user profile
    print("\nUser Profile:")
    profile = gpt.get_user_profile(user_id)
    for category, items in profile.items():
        if items and category != "user_id" and category != "total_memories":
            print(f"  {category.title()}: {items}")
    
    # Show memory stats
    print(f"\nMemory Stats:")
    stats = gpt.memory_agent.get_memory_stats()
    print(f"  Total memories: {stats.get('total_memories', 0)}")
    print(f"  Memory types: {stats.get('memory_types', {})}")
    print(f"  Unique users: {stats.get('unique_users', 0)}")


if __name__ == "__main__":
    main()