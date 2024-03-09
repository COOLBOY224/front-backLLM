from typing import List
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import  Document, Indexed, init_beanie
from enum import Enum
from openai import OpenAI
from fastapi.openapi.utils import get_openapi
import asyncio

# MongoDB settings
MONGO_URI = AsyncIOMotorClient("mongodb://localhost:27017")
MONGO_DB = "conversation_history"

# Initialize MongoDB client
def get_database_client() -> AsyncIOMotorClient:
    return AsyncIOMotorClient("mongodb://localhost:27017")

# Pydantic model for conversation creation
class ConversationCreate(BaseModel):
    query: str
    response: str
    state: str

# Pydantic model for conversation update
class ConversationUpdate(BaseModel):
    state: str

# Model for conversation
class Conversation(Document):
    class State(str, Enum):
        completed = "completed"
        ongoing = "ongoing"

    query: str
    response: str
    state: State

async def initialize_beanie(client: AsyncIOMotorClient):
    await init_beanie(database=client[MONGO_DB], document_models=[Conversation])

# OpenAI API key
OPENAI_API_KEY = "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"

# Initialize FastAPI app
app = FastAPI()

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="LLM Interaction API",
        version="2.0.0",
        description="This document describes the LLM Interaction API",
        routes=app.routes,
    )

    # Define responses
    responses = {
        '201': {
            'description': 'Successfully created resource with ID',
            'content': {
                'application/vnd.launchpad.v1+json': {
                    'schema': {
                        'description': 'Generated resource ID',
                        'type': 'object',
                        'properties': {
                            'id': {
                                'description': 'Unique ID of the resource',
                                'type': 'string',
                                'format': 'uuid'
                            }
                        },
                        'required': ['id']
                    }
                }
            }
        },
        'UpdatedResponse': {
            'description': 'Successfully updated specified resource'
        },
        'DeletedResponse': {
            'description': 'Successfully deleted specified resource(s)'
        },
        'InvalidParametersError': {
            'description': 'Invalid parameter(s)',
            'content': {
                'application/vnd.launchpad.v1+json': {
                    'schema': {
                        '$ref': '#/components/schemas/APIError'
                    },
                    'example': {
                        'code': 400,
                        'message': 'Invalid parameters provided'
                    }
                }
            }
        },
        'NotFoundError': {
            'description': 'Specified resource(s) was not found',
            'content': {
                'application/vnd.launchpad.v1+json': {
                    'schema': {
                        '$ref': '#/components/schemas/APIError'
                    },
                    'example': {
                        'code': 404,
                        'message': 'Specified resource(s) was not found'
                    }
                }
            }
        },
        'InvalidCreationError': {
            'description': 'Unable to create resource due to errors',
            'content': {
                'application/vnd.launchpad.v1+json': {
                    'schema': {
                        '$ref': '#/components/schemas/APIError'
                    },
                    'example': {
                        'code': 422,
                        'message': 'Unable to create resource'
                    }
                }
            }
        },
        'InternalServerError': {
            'description': 'Internal server error',
            'content': {
                'application/vnd.launchpad.v1+json': {
                    'schema': {
                        '$ref': '#/components/schemas/APIError'
                    },
                    'example': {
                        'code': 500,
                        'message': 'Internal server error'
                    }
                }
            }
        }
    }

    openapi_schema['paths']['/conversations/']['post']['responses'] = responses
    openapi_schema['paths']['/conversations/{conversation_id}']['put']['responses'] = responses
    openapi_schema['paths']['/conversations/{conversation_id}']['delete']['responses'] = responses
    openapi_schema['paths']['/prompt/']['post']['responses'] = responses

    app.openapi_schema = openapi_schema
    return app.openapi_schema

# Add the OpenAPI schema to the app
app.openapi = custom_openapi

# Initialize OpenAI API client
openai = OpenAI(api_key=OPENAI_API_KEY)

@app.on_event("startup")
async def startup_event():
    await initialize_beanie(client)

# Create conversation
@app.post("/conversations/", response_model=Conversation)
async def create_conversation(conversation_data: ConversationCreate, client: AsyncIOMotorClient = Depends(get_database_client)):
    await initialize_beanie(client)
    conversation = Conversation(**conversation_data.dict())
    await conversation.insert()
    return conversation


# Get all conversations
@app.get("/conversations/", response_model=List[Conversation])
async def get_all_conversations(client: AsyncIOMotorClient = Depends(get_database_client)):
    conversations = await Conversation.find_all().to_list()
    return conversations


# Get conversation by ID
@app.get("/conversations/{conversation_id}", response_model=Conversation)
async def get_conversation(conversation_id: str, client: AsyncIOMotorClient = Depends(get_database_client)):
    conversation = await Conversation.get(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


# Update conversation state
@app.put("/conversations/{conversation_id}", response_model=Conversation)
async def update_conversation_state(
    conversation_id: str, conversation_data: ConversationUpdate, client: AsyncIOMotorClient = Depends(get_database_client)
):
    conversation = await Conversation.get(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    if conversation_data.state not in [state.value for state in Conversation.State]:
        raise HTTPException(status_code=400, detail="Invalid state value")
    update_data = {"$set": {"state": conversation_data.state}}  # Modify the update_data format
    await conversation.update(update_data)
    return conversation


# Delete conversation
@app.delete("/conversations/{conversation_id}", response_model=dict)
async def delete_conversation(conversation_id: str, client: AsyncIOMotorClient = Depends(get_database_client)):
    conversation = await Conversation.get(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    await conversation.delete()
    return {"message": "Conversation deleted"}


# Send prompt query and receive response
@app.post("/prompt/")
async def prompt(prompt: str, max_tokens: int = 50, client: AsyncIOMotorClient = Depends(get_database_client)):
    # Get all conversations
    conversations = await Conversation.find_all().to_list()

    # Create context for prompt
    context = ""
    for conversation in conversations:
        context += f"User: {conversation.query}\nAI: {conversation.response}\n"

    # Send prompt to OpenAI
    response = openai.completions.create(
        model="gpt-3.5-turbo-16k",
        prompt=f"{context}User: {prompt}\nAI:",
        max_tokens=max_tokens,
    )

    # Store anonymized response in MongoDB
    conversation = Conversation(query=prompt, response=response.choices[0].text.strip(), state="completed")
    await conversation.insert()

    # Return anonymized response
    return {"response": conversation.response}

if __name__ == "__main__":
    import uvicorn

    client = get_database_client()
    uvicorn.run(app, host="localhost", port=8000)
    