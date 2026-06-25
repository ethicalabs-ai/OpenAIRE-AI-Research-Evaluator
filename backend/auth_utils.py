import datetime
import os

from fastapi import HTTPException, Request
from jose import JWTError, jwt

# Load .env early — this module is imported first by server.py
try:
    from dotenv import load_dotenv

    load_dotenv(
        dotenv_path=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"
        ),
        override=False,  # Never override values already set in the real environment
    )
except ImportError:
    pass  # python-dotenv optional; rely on OS env vars in production

# JWT signing secret — must be set in .env or host environment
_raw_secret = os.getenv("SECRET_KEY")
if not _raw_secret:
    import warnings

    warnings.warn(
        "SECRET_KEY is not set. Using an insecure development default. "
        "Set SECRET_KEY in your .env file before deploying.",
        stacklevel=1,
    )
    _raw_secret = "INSECURE-dev-only-change-me"

SECRET_KEY: str = _raw_secret
ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
COOKIE_NAME: str = "session_token"


def create_session_token(user_id: str, name: str) -> str:
    expire = datetime.datetime.utcnow() + datetime.timedelta(days=7)
    payload = {"sub": user_id, "name": name, "exp": expire}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(request: Request):
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        # Support Bearer token header as fallback
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
        else:
            return None
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        name: str = payload.get("name")
        if user_id is None:
            return None
        return {"id": user_id, "name": name}
    except JWTError:
        return None


def require_current_user(request: Request):
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user
