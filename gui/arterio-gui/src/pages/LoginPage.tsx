import { Link } from 'react-router'

export const LoginPage = () => {
    return (
        <div>
            <h1>Logowanie</h1>
            <p>
                Nie masz konta? <Link to="/register">Zarejestruj się</Link>
            </p>
        </div>
    )
}