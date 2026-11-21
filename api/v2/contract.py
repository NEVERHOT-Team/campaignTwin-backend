"""The seller service validates payloads and serves its own versioned OpenAPI document."""

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
    source = files("docs").joinpath("campaigntwin-api-v2.openapi.yaml")
    return yaml.safe_load(source.read_text(encoding="utf-8"))


@lru_cache
def validator(name):
    return Draft202012Validator(
        {"$ref": f"#/components/schemas/{name}", "components": contract()["components"]},
        format_checker=FormatChecker(),
    )


class V2Error(Exception):
    def __init__(self, status_code, code, message, **fields):
        self.status_code = status_code
        self.body = {"code": code, "message": message, **fields}


def _pointer(parts):
    return "".join("/" + str(key).replace("~", "~0").replace("/", "~1") for key in parts)


def validate_request(name, payload):
    try:
        json.dumps(payload, allow_nan=False)
    except ValueError as error:
        raise V2Error(422, "invalid_request", "Numeric fields must be finite.") from error
    issues = [
        {"field": _pointer(e.absolute_path), "message": e.message}
        for e in validator(name).iter_errors(payload)
    ]
    if issues:
        raise V2Error(422, "invalid_request", "Invalid request fields.", issues=issues)


def configure(app):
    @app.exception_handler(V2Error)
    async def domain_error(request: Request, exc: V2Error):
        return JSONResponse(exc.body, status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def invalid_input(request: Request, exc: RequestValidationError):
        issues = [{"field": _pointer(e["loc"][1:]), "message": e["msg"]} for e in exc.errors()]
        return JSONResponse(
            {"code": "invalid_request", "message": "Invalid request.", "issues": issues},
            status_code=422,
        )

    app.openapi = contract
