import { Link } from 'react-router'

export const RegisterPage = () => {
    return (
        <div>
            <h1>Rejestracja</h1>
            <p>
                Masz juz konto? <Link to="/login">Zaloguj sie</Link>
            </p>
        </div>
    )
}