import psycopg
from psycopg.rows import dict_row

from .config import DATABASE_URL


def connect():
    return psycopg.connect(DATABASE_URL, row_factory=dict_row, connect_timeout=5)
