from fastapi import APIRouter

from app.api.v1 import applications, catalog, contacts, ingest, matching, me, resumes

api_router = APIRouter()
api_router.include_router(me.router)
api_router.include_router(applications.router)
api_router.include_router(contacts.router)
api_router.include_router(catalog.router)
api_router.include_router(ingest.router)
api_router.include_router(resumes.router)
api_router.include_router(matching.router)
