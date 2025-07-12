
import json
from app.main import app

openapi_schema = app.openapi()

with open("docs/openapi.json", "w") as f:
    json.dump(openapi_schema, f, indent=2)
