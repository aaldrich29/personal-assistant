import os
import shutil
from typing import List, Optional, Literal
from fastapi import FastAPI, HTTPException, Body, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI()

VAULT_PATH = "/vault"

class FileContent(BaseModel):
    content: str

class RenameRequest(BaseModel):
    old_path: str
    new_path: str

class CreateRequest(BaseModel):
    path: str
    type: Literal["file", "folder"]

def get_safe_path(path: str):
    # Remove leading slashes/dots to keep it relative to vault
    clean_path = path.lstrip("/").lstrip(".")
    full_path = os.path.abspath(os.path.join(VAULT_PATH, clean_path))
    if not full_path.startswith(os.path.abspath(VAULT_PATH)):
        raise HTTPException(status_code=403, detail="Access denied")
    return full_path

def build_file_tree(path: str):
    name = os.path.basename(path)
    # Root case
    if path == VAULT_PATH:
        name = "root"
    
    item = {
        "name": name,
        "path": os.path.relpath(path, VAULT_PATH),
        "type": "folder" if os.path.isdir(path) else "file",
        "children": []
    }
    
    if item["path"] == ".":
        item["path"] = ""

    if os.path.isdir(path):
        try:
            # Sort: folders first, then files
            entries = os.listdir(path)
            entries.sort(key=lambda s: (not os.path.isdir(os.path.join(path, s)), s.lower()))
            
            for entry in entries:
                if entry.startswith("."): continue
                full_entry_path = os.path.join(path, entry)
                item["children"].append(build_file_tree(full_entry_path))
        except PermissionError:
            pass # Skip folders we can't read
            
    return item

@app.get("/")
async def read_root():
    return FileResponse("static/index.html")

@app.get("/api/tree")
async def get_tree():
    return build_file_tree(VAULT_PATH)

@app.get("/api/file/{file_path:path}")
async def read_file(file_path: str):
    full_path = get_safe_path(file_path)
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="File not found")
    if os.path.isdir(full_path):
        raise HTTPException(status_code=400, detail="Cannot read directory content as file")
    
    with open(full_path, "r", encoding="utf-8") as f:
        content = f.read()
    return {"content": content}

@app.post("/api/file/{file_path:path}")
async def save_file(file_path: str, file: FileContent):
    full_path = get_safe_path(file_path)
    try:
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(file.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"status": "success"}

@app.post("/api/create")
async def create_item(req: CreateRequest):
    full_path = get_safe_path(req.path)
    
    if os.path.exists(full_path):
        raise HTTPException(status_code=400, detail="Item already exists")
        
    try:
        if req.type == "folder":
            os.makedirs(full_path)
        else:
            # Ensure parent dir exists
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w") as f:
                f.write("")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"status": "success"}

@app.post("/api/rename")
async def rename_item(req: RenameRequest):
    old_full = get_safe_path(req.old_path)
    new_full = get_safe_path(req.new_path)
    
    if not os.path.exists(old_full):
        raise HTTPException(status_code=404, detail="Source not found")
    if os.path.exists(new_full):
        raise HTTPException(status_code=400, detail="Destination already exists")
        
    try:
        shutil.move(old_full, new_full)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"status": "success"}

@app.delete("/api/delete/{file_path:path}")
async def delete_item(file_path: str):
    full_path = get_safe_path(file_path)
    
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="Not found")
        
    try:
        if os.path.isdir(full_path):
            shutil.rmtree(full_path)
        else:
            os.remove(full_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"status": "success"}

app.mount("/static", StaticFiles(directory="static"), name="static")