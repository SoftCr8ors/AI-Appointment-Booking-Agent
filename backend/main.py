
"""
FastAPI backend for AI Appointment Booking Agent

Endpoints:
- POST /chat - Main conversation endpoint
- GET /health - Health check and uptime monitoring
- GET /debug/state/{session_id} - Debug endpoint to inspect session state

Features:
- Comprehensive error handling
- Request/response logging
- CORS support for frontend
- Session management with LangGraph
- Graceful degradation on service failures
"""
import logging
import os
import secrets
import time
import traceback
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from . import agent, config

# =============================================================================
# Logging Setup
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Reduce noise from external libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("google").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


# =============================================================================
# API key protection (optional)
# =============================================================================
# If API_KEY is set in the environment, /chat requires it in the X-API-Key header.
# If it is empty (local development), the endpoint stays open.
API_KEY = os.getenv("API_KEY", "").strip()


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Reject chat requests that do not carry the correct API key."""
    if API_KEY and not secrets.compare_digest(
        (x_api_key or "").encode(), API_KEY.encode()
    ):
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


# =============================================================================
# Application State
# =============================================================================
app_state = {
    "start_time": None,
    "requests_count": 0,
    "errors_count": 0,
    "graph_ready": False,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown logic"""
    # Startup
    logger.info("=" * 60)
    logger.info("AI Appointment Booking Agent - Starting Up")
    logger.info("=" * 60)

    app_state["start_time"] = time.time()

    try:
        # Pre-load the graph to catch configuration errors early
        logger.info("Initializing LangGraph agent...")
        agent.get_graph()
        app_state["graph_ready"] = True
        logger.info("✓ LangGraph agent ready")
    except Exception as e:
        logger.error(f"✗ Failed to initialize agent: {e}")
        logger.warning("API will start but chat endpoint will fail until fixed")

    logger.info(f"✓ Gemini Model: {config.GEMINI_MODEL}")
    logger.info(f"✓ Calendar ID: {config.CALENDAR_ID[:20]}...")
    logger.info(f"✓ Timezone: {config.TIMEZONE}")
    logger.info(
        f"✓ Business Hours: {config.BUSINESS_START} - {config.BUSINESS_END}"
    )
    logger.info(f"✓ Services: {', '.join(config.SERVICES)}")
    logger.info("=" * 60)
    logger.info("FastAPI server ready to accept requests")
    logger.info("=" * 60)

    yield

    # Shutdown
    logger.info("Shutting down gracefully...")
    uptime = time.time() - app_state["start_time"]
    logger.info(f"Total uptime: {uptime:.2f}s")
    logger.info(f"Total requests: {app_state['requests_count']}")
    logger.info(f"Total errors: {app_state['errors_count']}")


app = FastAPI(
    title="AI Appointment Booking Agent",
    description="LangGraph-based conversational AI for managing appointments",
    version="1.0.0",
    lifespan=lifespan,
)


# =============================================================================
# CORS Configuration
# =============================================================================
# Allow frontend from different origins (development and production)
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:5173,http://localhost:8000",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS + ["*"]
    if os.getenv("ENV") == "development"
    else ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)


# =============================================================================
# Request/Response Models
# =============================================================================
class ChatRequest(BaseModel):
    """Request payload for /chat endpoint"""

    session_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Unique identifier for the conversation session",
        examples=["user-123", "session-abc-def"],
    )
    message: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="User's message text",
        examples=["I want to book an appointment"],
    )

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, v: str) -> str:
        """Ensure session_id contains only safe characters"""
        if not v.replace("-", "").replace("_", "").isalnum():
            raise ValueError(
                "session_id must contain only letters, numbers, hyphens, and underscores"
            )
        return v.strip()

    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        """Trim whitespace from message"""
        stripped = v.strip()
        if not stripped:
            raise ValueError("message cannot be empty or only whitespace")
        return stripped


class ChatResponse(BaseModel):
    """Response payload for /chat endpoint"""

    reply: str = Field(..., description="AI assistant's response")
    session_id: str = Field(..., description="Session identifier echoed back")
    timestamp: str = Field(..., description="ISO 8601 timestamp of response")
    processing_time_ms: int = Field(
        ..., description="Time taken to process request in milliseconds"
    )


class HealthResponse(BaseModel):
    """Response payload for /health endpoint"""

    status: str = Field(..., description="Service health status")
    uptime_seconds: float = Field(
        ..., description="Time since server started"
    )
    timestamp: str = Field(..., description="Current server timestamp")
    graph_ready: bool = Field(
        ..., description="Whether LangGraph agent is initialized"
    )
    requests_processed: int = Field(
        ..., description="Total number of requests processed"
    )
    errors_encountered: int = Field(
        ..., description="Total number of errors encountered"
    )


class ErrorResponse(BaseModel):
    """Standard error response"""

    error: str = Field(..., description="Error type or category")
    message: str = Field(..., description="Human-readable error message")
    session_id: str | None = Field(None, description="Session ID if available")
    timestamp: str = Field(..., description="ISO 8601 timestamp of error")


# =============================================================================
# Middleware
# =============================================================================
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests and their processing time"""
    start_time = time.time()
    request_id = f"{int(start_time * 1000)}"

    # Log incoming request
    logger.info(f"→ [{request_id}] {request.method} {request.url.path}")

    try:
        response = await call_next(request)
        processing_time = (time.time() - start_time) * 1000

        # Log response
        logger.info(
            f"← [{request_id}] {response.status_code} "
            f"in {processing_time:.2f}ms"
        )

        # Add custom headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Processing-Time-Ms"] = str(int(processing_time))

        app_state["requests_count"] += 1
        return response

    except Exception as e:
        processing_time = (time.time() - start_time) * 1000
        logger.error(
            f"✗ [{request_id}] Error after {processing_time:.2f}ms: {str(e)}"
        )
        app_state["errors_count"] += 1
        raise


# =============================================================================
# Exception Handlers
# =============================================================================
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with consistent error format"""
    logger.warning(f"HTTP {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=f"http_{exc.status_code}",
            message=exc.detail,
            timestamp=datetime.utcnow().isoformat() + "Z",
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions gracefully"""
    logger.error(f"Unhandled exception: {exc}")
    logger.error(traceback.format_exc())

    app_state["errors_count"] += 1

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="internal_server_error",
            message="An unexpected error occurred. Please try again.",
            timestamp=datetime.utcnow().isoformat() + "Z",
        ).model_dump(),
    )


# =============================================================================
# API Endpoints
# =============================================================================
@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["monitoring"],
    summary="Health check endpoint",
    description="Returns service health status and metrics. Use for uptime monitoring and warmup.",
)
async def health_check():
    """
    Health check endpoint for monitoring and keeping the service warm.

    Returns current status, uptime, and basic metrics.
    """
    uptime = (
        time.time() - app_state["start_time"]
        if app_state["start_time"]
        else 0
    )

    return HealthResponse(
        status="healthy" if app_state["graph_ready"] else "degraded",
        uptime_seconds=round(uptime, 2),
        timestamp=datetime.utcnow().isoformat() + "Z",
        graph_ready=app_state["graph_ready"],
        requests_processed=app_state["requests_count"],
        errors_encountered=app_state["errors_count"],
    )


@app.post(
    "/chat",
    response_model=ChatResponse,
    tags=["chat"],
    dependencies=[Depends(require_api_key)],
    summary="Send a message to the AI assistant",
    description="Main conversation endpoint. Maintains conversation state using session_id.",
    responses={
        200: {"description": "Successful response from AI assistant"},
        400: {
            "description": "Invalid request (bad session_id or message)",
            "model": ErrorResponse,
        },
        500: {
            "description": "Internal server error",
            "model": ErrorResponse,
        },
        503: {
            "description": "Service unavailable (agent not ready)",
            "model": ErrorResponse,
        },
    },
)
def chat(request: ChatRequest):
    """
    Process a user message and return the AI assistant's response.

    The session_id is used as LangGraph's thread_id to maintain conversation context.
    Each session maintains its own conversation history and state.

    Args:
        request: ChatRequest containing session_id and message

    Returns:
        ChatResponse with the assistant's reply

    Raises:
        HTTPException: On validation errors or service failures
    """
    start_time = time.time()

    # Check if agent is ready
    if not app_state["graph_ready"]:
        logger.error("Chat request received but agent is not ready")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI service is not ready. Please try again in a moment.",
        )

    logger.info(
        f"Processing chat for session={request.session_id}, "
        f"message_length={len(request.message)}"
    )

    try:
        # Call the agent
        reply = agent.chat(
            session_id=request.session_id,
            message=request.message,
            graph=None,  # Uses cached graph
        )

        processing_time_ms = int((time.time() - start_time) * 1000)

        logger.info(
            f"Chat completed for session={request.session_id}, "
            f"reply_length={len(reply)}, "
            f"time={processing_time_ms}ms"
        )

        return ChatResponse(
            reply=reply,
            session_id=request.session_id,
            timestamp=datetime.utcnow().isoformat() + "Z",
            processing_time_ms=processing_time_ms,
        )

    except Exception as e:
        logger.error(f"Error in chat for session={request.session_id}: {e}")
        logger.error(traceback.format_exc())

        # Return a user-friendly error message
        # The actual error is logged but not exposed to the user
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "I'm having trouble processing your request right now. "
                "This could be due to a temporary issue with the AI service or calendar. "
                "Please try again in a moment."
            ),
        )


@app.get(
    "/debug/state/{session_id}",
    tags=["debug"],
    summary="Get session state (debug only)",
    description="Returns the internal state for a session. Use for debugging and testing.",
    include_in_schema=os.getenv("ENV") == "development",
)
async def debug_state(session_id: str):
    """
    Debug endpoint to inspect session state.

    Returns collected_info, pending_slot, and confirmed flag for a session.
    Only available in development mode.
    """
    if os.getenv("ENV") != "development":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Not found",
        )

    try:
        state = agent.debug_state(session_id)
        return {
            "session_id": session_id,
            "state": state,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
    except Exception as e:
        logger.error(
            f"Error getting state for session={session_id}: {e}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@app.get(
    "/",
    tags=["info"],
    summary="API information",
    description="Returns basic information about the API",
)
async def root():
    """Root endpoint with API information"""
    return {
        "name": "AI Appointment Booking Agent API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


# =============================================================================
# Development Server
# =============================================================================
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")

    logger.info(f"Starting development server on {host}:{port}")

    uvicorn.run(
        "backend.main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info",
    )

