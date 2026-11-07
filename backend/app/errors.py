class ApiError(Exception):
    def __init__(self, status, code, message, details=None, headers=None):
        self.status, self.code, self.message, self.details = status, code, message, details
        self.headers = headers or {}
