import {type FormEvent, useState} from "react";
import {useNavigate} from 'react-router';
import {useLogin} from "../hooks/useLogin.ts";
import styles from './LoginPage.module.css';

export const LoginPage = () => {
    const [username, setUsername] = useState("");
    const [password, setPassword] = useState("");
    const [showPassword, setShowPassword] = useState(false);

    const navigate = useNavigate();
    const loginMutation = useLogin();

    const handleSubmit = (e: FormEvent<HTMLFormElement>) => {
        e.preventDefault();

        if (loginMutation.isPending) {
            return;
        }

        loginMutation.mutate(
            {
                username: username.trim(),
                password: password,
            },
            {
                onSuccess: () => {
                    navigate("/", {replace: true});
                },
            },
        );
    }

    return (
        <div>
            <h1 className={styles.title}>Zaloguj się</h1>
            <p className={styles.subtitle}>
                Zaloguj się na konto otrzymane od administratora.
            </p>

            <form className={styles.form} onSubmit={handleSubmit} aria-busy={loginMutation.isPending}>
                <div className={styles.field}>
                    <label htmlFor="username">Identyfikator</label>
                    <input
                        id="username"
                        name="username"
                        type="text"
                        autoComplete="username"
                        autoCapitalize="none"
                        spellCheck={false}
                        placeholder="Twój identyfikator"
                        className={styles.input}
                        disabled={loginMutation.isPending}
                        required
                        value={username}
                        onChange={(e) => setUsername(e.target.value)}
                    />
                </div>

                <div className={styles.field}>
                    <label htmlFor="password">Hasło</label>
                    <div className={styles.passwordField}>
                        <input
                            id="password"
                            name="password"
                            type={showPassword ? 'text' : 'password'}
                            autoComplete="current-password"
                            placeholder="Twoje hasło"
                            className={`${styles.input} ${styles.passwordInput}`}
                            disabled={loginMutation.isPending}
                            required
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                        />
                        <button
                            type="button"
                            className={styles.passwordToggle}
                            aria-label={showPassword ? 'Ukryj hasło' : 'Pokaż hasło'}
                            aria-controls="password"
                            disabled={loginMutation.isPending}
                            onClick={() => setShowPassword(!showPassword)}
                        >
                            {showPassword ? 'Ukryj' : 'Pokaż'}
                        </button>
                    </div>
                </div>

                {loginMutation.isError && (
                    <p className={styles.error} role="alert">
                        {loginMutation.error.message}
                    </p>
                )}
                <button
                    type="submit"
                    className={styles.submit}
                    disabled={loginMutation.isPending}
                >
                    {loginMutation.isPending ? 'Logowanie…' : 'Zaloguj się'}
                </button>
            </form>
        </div>
    )
}
