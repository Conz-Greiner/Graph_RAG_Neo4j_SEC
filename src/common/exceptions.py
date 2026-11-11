class GraphRAGBaseException(Exception):
    """Base exception for all GraphRAG application errors."""
    pass

class ConfigurationError(GraphRAGBaseException):
    """Raised when there is a configuration issue."""
    pass

class LLMGatewayError(GraphRAGBaseException):
    """Raised when the LLM gateway encounters an error."""
    pass

class IngestionError(GraphRAGBaseException):
    """Raised during document ingestion or parsing failures."""
    pass

class RetrievalError(GraphRAGBaseException):
    """Raised during retrieval from vector or graph stores."""
    pass

class AgentError(GraphRAGBaseException):
    """Raised when the LangGraph agent encounters an unrecoverable error."""
    pass