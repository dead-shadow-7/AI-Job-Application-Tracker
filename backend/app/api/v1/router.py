from fastapi import APIRouter

from app.api.v1 import applications, catalog, contacts, me

api_router = APIRouter()
api_router.include_router(me.router)
api_router.include_router(applications.router)
api_router.include_router(contacts.router)
api_router.include_router(catalog.router)
