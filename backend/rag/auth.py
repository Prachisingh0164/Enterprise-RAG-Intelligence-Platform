import hashlib, hmac, os, datetime as dt, jwt
from dataclasses import dataclass
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from .config import SECRET

ALL_ROLES = ["all", "employee", "hr", "finance", "admin"]


def hash_pw(pw, salt=None):
    salt = salt or os.urandom(8).hex()
    return f"{salt}${hashlib.pbkdf2_hmac('sha256', pw.encode(), salt.encode(), 100_000).hex()}"


def verify_pw(pw, stored):
    return hmac.compare_digest(hash_pw(pw, stored.split("$")[0]), stored)


def roles_for(role):                                   # roles whose documents a user may read
    return list(ALL_ROLES) if role == "admin" else ["all", role]


def make_token(username, role):
    return jwt.encode({"sub": username, "role": role, "exp": dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=8)}, SECRET, algorithm="HS256")


@dataclass
class Principal:
    username: str
    role: str


def current_user(cred: HTTPAuthorizationCredentials = Depends(HTTPBearer())):
    try:
        p = jwt.decode(cred.credentials, SECRET, algorithms=["HS256"])
        return Principal(p["sub"], p["role"])
    except Exception:
        raise HTTPException(401, "Invalid or expired token")
