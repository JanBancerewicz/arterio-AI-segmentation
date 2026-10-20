import { Outlet } from 'react-router'
import { Disclaimer } from '../components/Disclaimer.tsx'
import styles from './AuthLayout.module.css'
import {Title} from "../components/Title.tsx";

export const AuthLayout = () => {
    return (
        <div className={styles.auth}>
            <main className={styles.main}>
                <div className={styles.card}>
                    <div className={styles.brand}>
                        <Title variant="dark" />
                    </div>
                    <Outlet />
                </div>
            </main>
            <footer><Disclaimer /></footer>
        </div>
    )
}
