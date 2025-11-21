
# from crewai.tools import BaseTool
# from typing import Type, Optional, List, Dict, Any
# from pydantic import BaseModel, Field
# import glob
# from crewai_tools import RagTool
# import os
# import traceback

# # Absolute paths
# knowledge_path = r"C:\self_correcting_rag\knowledge"
# db_path = r"C:\self_correcting_rag\db"


# def get_rag_config() -> Dict[str, Any]:
#     """Return RAG config using Google/Gemini embeddings + LanceDB vectordb."""
#     llm_model = os.getenv("LLM_MODEL", "gemini-1.5-flash")

#     gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
#     if not gemini_key:
#         raise ValueError("❌ Missing GEMINI_API_KEY environment variable")

#     os.environ["GOOGLE_API_KEY"] = gemini_key
#     os.makedirs(db_path, exist_ok=True)

#     print(f"📁 Vector DB path: {db_path}")
#     print(f"📁 Knowledge path: {knowledge_path}")

#     return {
#         "llm": {
#             "provider": "google",
#             "config": {
#                 "model": llm_model,
#                 "api_key": gemini_key
#             }
#         },
#         "embedder": {
#             "provider": "google",
#             "config": {
#                 "model": "models/text-embedding-004"
#             }
#         },
#         "vectordb": {
#             "provider": "lancedb",
#             "config": {
#                 "dir": db_path
#             }
#         },
#         "chunker": {
#             "chunk_size": 800,
#             "chunk_overlap": 100,
#             "min_chunk_size": 400
#         }
#     }


# class SearchInput(BaseModel):
#     """Input schema for search."""
#     query: str = Field(..., description="The search query")


# class KnowledgeBaseTool(BaseTool):
#     """Tool for semantic search over indexed documents."""
#     name: str = "search_knowledge_base"
#     description: str = (
#         "Search the knowledge base for information. "
#         "ALWAYS use this tool to find information from uploaded documents. "
#         "Input: a search query string. "
#         "Output: relevant information from the documents."
#     )
#     args_schema: Type[BaseModel] = SearchInput
#     rag_tool: Optional[RagTool] = None

#     def __init__(self):
#         super().__init__()
#         object.__setattr__(self, "rag_tool", self._init_rag())

#     def _init_rag(self) -> RagTool:
#         """Initialize RagTool and index documents."""
#         config = get_rag_config()

#         supported_extensions = ['.txt', '.pdf', '.docx', '.md', '.csv']
#         data_files: List[str] = []

#         if not os.path.exists(knowledge_path):
#             os.makedirs(knowledge_path, exist_ok=True)
#             raise ValueError(f"❌ Knowledge folder not found: {knowledge_path}")

#         # Find files
#         for ext in supported_extensions:
#             pattern = os.path.join(knowledge_path, f"**/*{ext}")
#             found = glob.glob(pattern, recursive=True)
#             data_files.extend(found)

#         data_files = [f for f in data_files if 'db' not in f and os.path.isfile(f)]

#         if not data_files:
#             raise ValueError(f"❌ No documents in {knowledge_path}")

#         print(f"\n📚 Found {len(data_files)} files:")
#         for i, file in enumerate(data_files, 1):
#             size = os.path.getsize(file) / 1024
#             print(f"   {i}. {os.path.basename(file)} ({size:.2f} KB)")

#         print(f"\n🔄 Starting indexing...")

#         try:
#             # Initialize RagTool first
#             rag_tool = RagTool(config=config)
            
#             # Manually add each file to ensure indexing
#             print("\n📝 Adding files to RAG:")
#             for file in data_files:
#                 try:
#                     print(f"   Adding: {os.path.basename(file)}...")
#                     rag_tool.add(file)
#                     print(f"   ✓ Added: {os.path.basename(file)}")
#                 except Exception as e:
#                     print(f"   ✗ Failed to add {file}: {e}")

#             # Verify indexing
#             try:
#                 import lancedb
#                 db = lancedb.connect(db_path)
#                 tables = db.table_names()
#                 print(f"\n📊 Vector DB verification:")
#                 print(f"   Tables: {tables}")
                
#                 if tables:
#                     table = db.open_table(tables[0])
#                     df = table.to_pandas()
#                     print(f"   Indexed chunks: {len(df)}")
#                     print(f"   Columns: {list(df.columns)}")
                    
#                     if len(df) == 0:
#                         print("\n   ⚠️ WARNING: No chunks were indexed!")
#                         print("   This means the files couldn't be processed.")
#                     else:
#                         # Show sample
#                         text_col = None
#                         for col in ['doc', 'content', 'text', 'chunk']:
#                             if col in df.columns:
#                                 text_col = col
#                                 break
                        
#                         if text_col:
#                             print(f"\n   Sample chunks from '{text_col}':")
#                             for i in range(min(3, len(df))):
#                                 sample = str(df.iloc[i][text_col])[:150]
#                                 print(f"   [{i+1}] {sample}...")
#                 else:
#                     print("   ⚠️ No tables created!")
#             except Exception as e:
#                 print(f"   ⚠️ Verification error: {e}")

#             print(f"\n✅ RAG tool ready\n")
#             return rag_tool

#         except Exception as e:
#             print(f"\n❌ Indexing error: {e}")
#             traceback.print_exc()
#             raise

#     def _run(self, query: str) -> str:
#         """Search the knowledge base."""
#         if not self.rag_tool:
#             return "❌ RAG tool not initialized"

#         try:
#             print(f"\n{'='*70}")
#             print(f"🔍 SEARCH: '{query}'")
#             print(f"{'='*70}")

#             # Try RagTool search
#             result = None
#             if hasattr(self.rag_tool, "_run"):
#                 try:
#                     result = self.rag_tool._run(query)
#                 except Exception as e:
#                     print(f"   RagTool._run error: {e}")

#             if hasattr(self.rag_tool, "search") and not result:
#                 try:
#                     result = self.rag_tool.search(query)
#                 except Exception as e:
#                     print(f"   RagTool.search error: {e}")

#             print(f"📊 RagTool returned: {len(result) if result else 0} chars")

#             # Fallback: Direct vector search
#             if not result or len(result) < 50:
#                 print("🔄 Direct vector search...")
                
#                 import lancedb
#                 import google.generativeai as genai

#                 gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
#                 genai.configure(api_key=gemini_key)

#                 embed_result = genai.embed_content(
#                     model="models/text-embedding-004",
#                     content=query,
#                     task_type="retrieval_query"
#                 )

#                 if isinstance(embed_result, dict) and 'embedding' in embed_result:
#                     query_embedding = embed_result['embedding']
#                 elif hasattr(embed_result, 'embedding'):
#                     query_embedding = embed_result.embedding
#                 else:
#                     query_embedding = embed_result['embeddings'][0]['values']

#                 db = lancedb.connect(db_path)
#                 tables = db.table_names()
                
#                 if not tables:
#                     return "❌ No vector DB tables found. Re-index required."

#                 table = db.open_table(tables[0])
#                 search_results = table.search(query_embedding).limit(10).to_pandas()
                
#                 print(f"   Vector search: {len(search_results)} results")

#                 if search_results.empty:
#                     return (
#                         f"No information found for: '{query}'\n\n"
#                         f"The knowledge base may not contain this information."
#                     )

#                 # Find text column
#                 text_col = None
#                 for col in ['doc', 'content', 'text', 'chunk', 'page_content']:
#                     if col in search_results.columns:
#                         text_col = col
#                         break

#                 if not text_col:
#                     return f"❌ No text column. Columns: {list(search_results.columns)}"

#                 # Collect results
#                 texts = []
#                 for idx, row in search_results.iterrows():
#                     text = str(row[text_col])
#                     if text and text != 'nan' and len(text) > 10:
#                         score = row.get('_distance', 'N/A')
#                         texts.append(f"[Relevance: {score}]\n{text}")

#                 if texts:
#                     result = "\n\n" + "="*70 + "\n\n".join(texts)
#                     print(f"✅ Retrieved {len(texts)} chunks")
#                 else:
#                     return f"No content found for: '{query}'"

#             if not result or len(result) < 30:
#                 return (
#                     f"⚠️ No relevant information for: '{query}'\n\n"
#                     f"Try different keywords or check if information exists in documents."
#                 )

#             print(f"✅ Result: {len(result)} chars")
#             print(f"   Preview: {result[:250]}...")
#             print(f"{'='*70}\n")
#             return result

#         except Exception as e:
#             print(f"❌ Search error: {e}")
#             traceback.print_exc()
#             return f"❌ Search error: {e}"


# _rag_tool_instance: Optional[KnowledgeBaseTool] = None


# def get_rag_tool() -> KnowledgeBaseTool:
#     """Get singleton RAG tool."""
#     global _rag_tool_instance
#     if _rag_tool_instance is None:
#         print("\n🔧 Initializing Knowledge Base...")
#         _rag_tool_instance = KnowledgeBaseTool()
#         print("✅ Knowledge Base ready!\n")
#     return _rag_tool_instance


from crewai.tools import BaseTool
from typing import Type, Optional, List, Dict, Any
from pydantic import BaseModel, Field
import glob
import os
from crewai_tools import RagTool

# --------------------------------------
# GLOBAL PATHS
# --------------------------------------
knowledge_path = r"C:\self_correcting_rag\knowledge"
db_path = r"C:\self_correcting_rag\db"


# --------------------------------------
# RAG CONFIG (Single clean version)
# --------------------------------------
def get_rag_config() -> Dict[str, Any]:
    """Return RAG configuration for Gemini + LanceDB."""
    llm_model = os.getenv("LLM_MODEL", "gemini-1.5-flash")
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    if not gemini_key:
        raise ValueError("❌ Missing GEMINI_API_KEY")

    os.environ["GOOGLE_API_KEY"] = gemini_key
    os.makedirs(db_path, exist_ok=True)

    return {
        "llm": {
            "provider": "google",
            "config": {
                "model": llm_model,
                "api_key": gemini_key,
            }
        },
        "embedder": {
            "provider": "google",
            "config": {
                "model": "models/text-embedding-004"
            }
        },
        "vectordb": {
            "provider": "lancedb",
            "config": {
                "dir": db_path
            }
        },
        "chunker": {
            "chunk_size": 800,
            "chunk_overlap": 100,
            "min_chunk_size": 300,
        }
    }


# --------------------------------------
# Input Schema
# --------------------------------------
class SearchInput(BaseModel):
    query: str = Field(..., description="Text query to search the knowledge base")


# --------------------------------------
# SINGLE KNOWLEDGE BASE TOOL
# --------------------------------------
class KnowledgeBaseTool(BaseTool):
    name: str = "search_knowledge_base"
    description: str = (
        "Search the knowledge base using RAG. "
        "Input: query string. Output: relevant text chunks."
    )
    args_schema: Type[BaseModel] = SearchInput
    rag_tool: Optional[RagTool] = None

    def __init__(self):
        super().__init__()
        object.__setattr__(self, "rag_tool", self._initialize_rag())

    # --------------------------------------
    # RAG INITIALIZATION
    # --------------------------------------
    def _initialize_rag(self) -> RagTool:
        """Initialize RagTool and index all documents (only once)."""
        config = get_rag_config()

        if not os.path.exists(knowledge_path):
            os.makedirs(knowledge_path, exist_ok=True)
            raise ValueError(f"❌ Knowledge folder missing: {knowledge_path}")

        supported_ext = ['.txt', '.pdf', '.docx', '.md', '.csv']
        files = []

        # collect files
        for ext in supported_ext:
            files.extend(glob.glob(os.path.join(knowledge_path, f"**/*{ext}"), recursive=True))

        files = [f for f in files if os.path.isfile(f) and 'db' not in f]

        if not files:
            raise ValueError(f"❌ No documents found in: {knowledge_path}")

        print(f"\n📚 Found {len(files)} documents, indexing...")

        rag = RagTool(config=config)

        for file in files:
            try:
                print(f"   ➜ Adding {os.path.basename(file)}")
                rag.add(file)
            except Exception as e:
                print(f"   ❌ Failed on {file}: {e}")

        print("✅ RAG indexing completed.\n")
        return rag

    # --------------------------------------
    # SEARCH
    # --------------------------------------
    def _run(self, query: str) -> str:
        """Search through RAG."""
        if not self.rag_tool:
            return "❌ RAG tool not initialized"

        try:
            print(f"\n🔎 Searching: {query}")

            # clean single search call
            result = self.rag_tool._run(query)

            if not result or len(result) < 10:
                return f"No relevant information found for: '{query}'"

            return result

        except Exception as e:
            return f"❌ Search error: {e}"



_rag_tool_instance: Optional[KnowledgeBaseTool] = None


def get_rag_tool() -> KnowledgeBaseTool:
    """Singleton RAG tool initialization."""
    global _rag_tool_instance
    if _rag_tool_instance is None:
        print("\n🔧 Initializing RAG tool...")
        _rag_tool_instance = KnowledgeBaseTool()
        print("✅ RAG ready!\n")
    return _rag_tool_instance
