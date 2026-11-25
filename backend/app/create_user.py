"""Usage: python -m app.create_user --username patryk --display-name 'Patryk'."""
import argparse
import getpass
import sys
import warnings

import psycopg

from .users import create_user, normalize_username


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Utwórz konto użytkownika (bez rejestracji przez API).")
    parser.add_argument("--username", required=True, help="Unikalny identyfikator, 3–64 znaki.")
    parser.add_argument("--display-name", required=True, help="Nazwa widoczna w aplikacji, 1–100 znaków.")
    args = parser.parse_args(argv)
    try:
        username = normalize_username(args.username)
        # Never fall back to echoing a password when there is no usable terminal.
        with warnings.catch_warnings():
            warnings.simplefilter("error", getpass.GetPassWarning)
            password = getpass.getpass("Hasło tymczasowe (12–128 znaków): ")
            confirmation = getpass.getpass("Powtórz hasło tymczasowe: ")
        if password != confirmation:
            raise ValueError("Podane hasła nie są identyczne.")
        user = create_user(username, args.display_name, password, must_change_password=True)
    except (EOFError, getpass.GetPassWarning):
        print("Nie można bezpiecznie odczytać hasła. Uruchom polecenie w interaktywnym terminalu.", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nAnulowano tworzenie konta.", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except psycopg.Error:
        print("Nie udało się zapisać konta. Sprawdź połączenie z bazą i wykonanie migracji.", file=sys.stderr)
        return 1
    print(f"Utworzono konto: {user.username} (id: {user.id}).")
    print("Użytkownik zmieni hasło tymczasowe przy pierwszym logowaniu.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
