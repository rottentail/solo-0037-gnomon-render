class RenderError(ValueError):
    def __init__(self, code, detail, status=422):
        super().__init__(detail)
        self.code = code
        self.detail = detail
        self.status = status


def require(condition, code, detail, status=422):
    if not condition:
        raise RenderError(code, detail, status)
