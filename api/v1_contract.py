"""Use the published v1 request contract for validation and FastAPI discovery."""

import json
from functools import lru_cache
from importlib.resources import files

import yaml
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from jsonschema import Draft202012Validator, FormatChecker


@lru_cache
def contract():
    source = files("docs").joinpath("campaigntwin-api-v1.openapi.yaml")
    return yaml.safe_load(source.read_text(encoding="utf-8"))


class V1Error(Exception):
    def __init__(self, status_code, code, message, **fields):
        self.status_code = status_code
        self.body = {"code": code, "message": message, **fields}


@lru_cache
def validator(name):
    return Draft202012Validator(
        {"$ref": f"#/components/schemas/{name}", "components": contract()["components"]},
        format_checker=FormatChecker(),
    )


def pointer(parts):
    return "".join("/" + str(key).replace("~", "~0").replace("/", "~1") for key in parts)


def validate_request(name, payload):
    try:
        json.dumps(payload, allow_nan=False)
    except ValueError as error:
        raise V1Error(422, "invalid_request", "Numeric fields must be finite.") from error
    errors = list(validator(name).iter_errors(payload))
    if errors:
        raise V1Error(422, "invalid_request", "Invalid request fields.", issues=[
            {"field": pointer(error.absolute_path), "message": error.message}
            for error in errors
        ])


def configure(app):
    from fastapi.exception_handlers import request_validation_exception_handler

    @app.exception_handler(V1Error)
    async def v1_error(request: Request, exc: V1Error):
        return JSONResponse(exc.body, status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def invalid_input(request: Request, exc: RequestValidationError):
        if not request.url.path.startswith("/api/v1/"):
            return await request_validation_exception_handler(request, exc)
        issues = [{"field": pointer(err["loc"][1:]), "message": err["msg"]} for err in exc.errors()]
        return JSONResponse({"code": "invalid_request", "message": "Invalid request.",
                             "issues": issues}, status_code=422)

    generated = app.openapi

    def openapi():
        schema = generated()
        public = contract()
        schema["paths"].update(public["paths"])
        for section, definitions in public["components"].items():
            target = schema["components"].setdefault(section, {})
            # Legacy routes use names such as RunStatus. Refuse ambiguous schema names.
            for name, definition in definitions.items():
                if name in target and target[name] != definition:
                    raise ValueError(f"OpenAPI component collision: {section}/{name}")
                target[name] = definition
        return schema

    app.openapi = openapi
