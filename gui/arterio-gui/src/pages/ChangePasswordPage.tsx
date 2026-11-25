import {type FormEvent, useState} from "react";
import {Link, useNavigate} from "react-router";
import {useSession} from "../hooks/useSession.ts";
import {useChangePassword} from "../hooks/useChangePassword.ts";
import {useLogout} from "../hooks/useLogout.ts";
import styles from './ChangePasswordPage.module.css';

const validatePasswords = (currentPassword: string, newPassword: string, confirmation: string) => {
    if (newPassword.length < 12 || newPassword.length > 128 || newPassword.trim() === "") {
        return "Nowe hasło musi mieć od 12 do 128 znaków i nie może składać się z samych spacji.";
    }
    if (newPassword === currentPassword) {
        return "Nowe hasło musi różnić się od obecnego.";
    }
    if (newPassword !== confirmation) {
        return "Podane nowe hasła nie są identyczne.";
    }
    return null;
}

export const ChangePasswordPage = () => {
    const [currentPassword, setCurrentPassword] = useState("");
    const [newPassword, setNewPassword] = useState("");
    const [confirmation, setConfirmation] = useState("");
    const [showPasswords, setShowPasswords] = useState(false);
    const [validationError, setValidationError] = useState<string | null>(null);

    const {data: user} = useSession();
    const changePasswordMutation = useChangePassword();
    const logoutMutation = useLogout();
    const navigate = useNavigate();

    const forced = user?.mustChangePassword === true;
    const busy = changePasswordMutation.isPending || logoutMutation.isPending;
    const passwordType = showPasswords ? 'text' : 'password';

    const handleSubmit = (e: FormEvent<HTMLFormElement>) => {
        e.preventDefault();

        if (busy) {
            return;
        }

        const error = validatePasswords(currentPassword, newPassword, confirmation);
        setValidationError(error);
        if (error) {
            return;
        }

        changePasswordMutation.mutate(
            {currentPassword, newPassword},
            {
                onSuccess: () => {
                    navigate("/", {replace: true});
                },
            },
        );
    }

    return (
        <div>
            <h1 className={styles.title}>{forced ? 'Ustaw własne hasło' : 'Zmień hasło'}</h1>
            <p className={styles.subtitle}>
                {forced
                    ? 'Konto ma hasło tymczasowe nadane przez administratora. Przed rozpoczęciem pracy ustaw hasło znane tylko Tobie.'
                    : 'Po zmianie hasła pozostałe sesje na tym koncie zostaną wylogowane.'}
            </p>

            <form className={styles.form} onSubmit={handleSubmit} aria-busy={busy}>
                <div className={styles.field}>
                    <label htmlFor="current-password">{forced ? 'Hasło tymczasowe' : 'Obecne hasło'}</label>
                    <input
                        id="current-password"
                        name="current-password"
                        type={passwordType}
                        autoComplete="current-password"
                        className={styles.input}
                        disabled={busy}
                        required
                        value={currentPassword}
                        onChange={(e) => setCurrentPassword(e.target.value)}
                    />
                </div>

                <div className={styles.field}>
                    <label htmlFor="new-password">Nowe hasło</label>
                    <input
                        id="new-password"
                        name="new-password"
                        type={passwordType}
                        autoComplete="new-password"
                        aria-describedby="new-password-hint"
                        className={styles.input}
                        disabled={busy}
                        required
                        value={newPassword}
                        onChange={(e) => setNewPassword(e.target.value)}
                    />
                    <span id="new-password-hint" className={styles.hint}>Od 12 do 128 znaków.</span>
                </div>

                <div className={styles.field}>
                    <label htmlFor="confirm-password">Powtórz nowe hasło</label>
                    <input
                        id="confirm-password"
                        name="confirm-password"
                        type={passwordType}
                        autoComplete="new-password"
                        className={styles.input}
                        disabled={busy}
                        required
                        value={confirmation}
                        onChange={(e) => setConfirmation(e.target.value)}
                    />
                </div>

                <label className={styles.checkbox}>
                    <input
                        type="checkbox"
                        checked={showPasswords}
                        disabled={busy}
                        onChange={(e) => setShowPasswords(e.target.checked)}
                    />
                    Pokaż hasła
                </label>

                {(validationError || changePasswordMutation.isError) && (
                    <p className={styles.error} role="alert">
                        {validationError ?? changePasswordMutation.error?.message}
                    </p>
                )}
                <button type="submit" className={styles.submit} disabled={busy}>
                    {changePasswordMutation.isPending ? 'Zapisywanie…' : 'Zmień hasło'}
                </button>
            </form>

            {forced ? (
                <button
                    type="button"
                    className={styles.secondary}
                    disabled={busy}
                    onClick={() => logoutMutation.mutate()}
                >
                    {logoutMutation.isPending ? 'Wylogowywanie…' : 'Wyloguj'}
                </button>
            ) : (
                <Link className={styles.secondary} to="/">Anuluj</Link>
            )}
            {logoutMutation.isError && (
                <p className={styles.error} role="alert">Nie udało się wylogować. Spróbuj ponownie.</p>
            )}
        </div>
    )
}
