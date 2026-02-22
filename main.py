import os
import chromadb
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Please set it in your environment or .env file.")

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

client = chromadb.PersistentClient(path="recipe_db")
collection = client.get_or_create_collection(name="recipes")


app = FastAPI(
    title="Chef Gemini RAG API",
    description="A RAG-based recipe recommender using ChromaDB and Gemini 1.5 Flash."
)


class RecipeRequest(BaseModel):
    ingredients: List[str]
    constraints: Optional[str] = "None"


def format_with_gemini(available_ingredients, recipes_data, constraints):
    """
    Constructs the prompt and gets the response from Gemini.
    """
    
    context_text = ""
    for i, doc in enumerate(recipes_data['documents']):
        meta = recipes_data['metadatas'][i]
        context_text += f"\n--- OPTION {i+1}: {meta.get('RecipeName', 'Unknown Recipe')} ---\n"
        context_text += f"Cooking Time: {meta.get('TotalTimeInMins', 'N/A')} mins | Servings: {meta.get('Servings', 'N/A')}\n"
        context_text += f"Instructions: {doc}\n"
    
   
    prompt = (
        f"You are a friendly, expert kitchen assistant. A home cook has: {', '.join(available_ingredients)}.\n"
        f"IMPORTANT DIETARY/EQUIPMENT CONSTRAINTS: {constraints}\n\n"
        f"Based on their ingredients AND the constraints above, suggest the 3 best matches from the provided database. "
        f"If a recipe uses an ingredient they are allergic to or requires equipment they don't have, "
        f"you MUST modify the steps to accommodate them or explain why a certain substitution was made.\n\n"
        f"Don't mention 'searching a database'—just offer your best culinary advice based on these recipes:\n"
        f"{context_text}"
    )

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating content: {str(e)}"


@app.get("/")
async def health_check():
    """Verifies the API and Database are online."""
    return {
        "status": "online",
        "database_count": collection.count(),
        "model": "gemini-1.5-flash"
    }

@app.post("/recommend")
async def recommend_recipes(request: RecipeRequest):
    """
    Takes ingredients and constraints, queries ChromaDB, and returns a Gemini summary.
    """
    try:
        
        query_string = ", ".join(request.ingredients)
        results = collection.query(
            query_texts=[query_string],
            n_results=3,
            include=["documents", "metadatas"]
        )

        
        if not results["documents"] or not results["documents"][0]:
            return {"recommendation": "I'm sorry, I couldn't find any recipes that match those ingredients in my current database."}

       
        recipes_for_gemini = {
            "documents": results["documents"][0],
            "metadatas": results["metadatas"][0]
        }

        
        final_answer = format_with_gemini(
            request.ingredients, 
            recipes_for_gemini, 
            request.constraints
        )

        return {
            "status": "success",
            "ingredients_received": request.ingredients,
            "constraints_applied": request.constraints,
            "recommendation": final_answer
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

