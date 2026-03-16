import chromadb
from chromadb.config import Settings
from typing import Dict, List, Optional
from pathlib import Path

def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """Discover available ChromaDB backends in the project directory"""
    backends = {}
    current_dir = Path(".")
    
    # Look for ChromaDB directories
    chroma_dirs = [d for d in current_dir.iterdir() if d.is_dir() and 'chroma' in d.name.lower()]

    # Loop through each discovered directory
    for dir_path in chroma_dirs:
        # Wrap connection attempt in try-except block for error handling
        try:
            
            # Initialize database client with directory path and configuration settings
            client = chromadb.PersistentClient(path=str(dir_path))
            
            # Retrieve list of available collections from the database
            collections = client.list_collections()
            
            # Loop through each collection found
            for coll in collections:
                # Create unique identifier key combining directory and collection names
                key = f"{dir_path.name}_{coll.name}"
                # Build information dictionary containing:
                # Store directory path as string
                # Store collection name
                # Create user-friendly display name
                # Get document count with fallback for unsupported operations
                info = {
                    "directory": str(dir_path),
                    "collection_name": coll.name,
                    "display_name": f"{dir_path.name} - {coll.name}",
                    "count": str(coll.count()) if hasattr(coll, 'count') else "Unknown"
                }
                # Add collection information to backends dictionary
                backends[key] = info
        
        # Handle connection or access errors gracefully
        except Exception as e:
            # Create fallback entry for inaccessible directories
            key = f"{dir_path.name}_error"
            # Include error information in display name with truncation
            # Set appropriate fallback values for missing information
            backends[key] = {
                "directory": str(dir_path),
                "collection_name": "N/A",
                "display_name": f"{dir_path.name} (Error: {str(e)[:50]}...)",
                "count": "N/A"
            }

    # Return complete backends dictionary with all discovered collections
    return backends

def initialize_rag_system(chroma_dir: str, collection_name: str):
    """Initialize the RAG system with specified backend (cached for performance)"""
    try:
    # Create a chromadb persistent client
        client = chromadb.PersistentClient(path=chroma_dir)
        # Return the collection with the collection_name
        collection = client.get_collection(name=collection_name)
        return collection, True, None
    except Exception as e:
        print(f"Error initializing RAG system: {e}")
        return None, False, str(e)

def retrieve_documents(collection, query: str, n_results: int = 3, 
                      mission_filter: Optional[str] = None) -> Optional[Dict]:
    """Retrieve relevant documents from ChromaDB with optional filtering"""

    # TODO: Initialize filter variable to None (represents no filtering)
    filter_dict = None
    # TODO: Check if filter parameter exists and is not set to "all" or equivalent
    if mission_filter and mission_filter.lower() != "all":
        # TODO: If filter conditions are met, create filter dictionary with appropriate field-value pairs
        filter_dict = {"mission": mission_filter}

    # TODO: Execute database query with the following parameters:
        # TODO: Pass search query in the required format
        # TODO: Set maximum number of results to return
        # TODO: Apply conditional filter (None for no filtering, dictionary for specific filtering)
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=filter_dict
    )
    # TODO: Return query results to caller
    return results

def format_context(documents: List[str], metadatas: List[Dict]) -> str:
    """Format retrieved documents into context"""
    if not documents:
        return ""
    
    # TODO: Initialize list with header text for context section
    context_parts = ["Context from retrieved documents:"]
    # TODO: Loop through paired documents and their metadata using enumeration
        # TODO: Extract mission information from metadata with fallback value
        # TODO: Clean up mission name formatting (replace underscores, capitalize)
        # TODO: Extract source information from metadata with fallback value  
        # TODO: Extract category information from metadata with fallback value
        # TODO: Clean up category name formatting (replace underscores, capitalize)
    for idx, (doc, meta) in enumerate(zip(documents, metadatas)):
        mission = meta.get("mission", "Unknown Mission").replace("_", " ").title()
        source = meta.get("source", "Unknown Source")
        category = meta.get("category", "Unknown Category").replace("_", " ").title()    
        # TODO: Create formatted source header with index number and extracted information
        header = f"Document {idx+1}: {mission} - {category} (Source: {source})"
        # TODO: Add source header to context parts list
        context_parts.append(header)
        # TODO: Check document length and truncate if necessary
        if len(doc) > 1000:
            doc = doc[:1000] + "\n\n[Document truncated due to length]"
        # TODO: Add truncated or full document content to context parts list
        context_parts.append(doc)

    # TODO: Join all context parts with newlines and return formatted string
    return "\n\n".join(context_parts)