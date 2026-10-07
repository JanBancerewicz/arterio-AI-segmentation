import { Outlet } from 'react-router'
import { Disclaimer } from '../components/Disclaimer.tsx'
import styles from './AuthLayout.module.css'

export const AuthLayout = () => {
    return (
        <div className={styles.auth}>
            <main className={styles.card}><Outlet /></main>
            <footer><Disclaimer /></footer>
        </div>
    )
}