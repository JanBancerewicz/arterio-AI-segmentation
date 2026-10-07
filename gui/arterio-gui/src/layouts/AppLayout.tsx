import { Outlet } from 'react-router'
import { Sidebar } from '../components/Sidebar.tsx'
import { Disclaimer } from '../components/Disclaimer.tsx'
import styles from './AppLayout.module.css'

export const AppLayout = () => {
    return (
        <div className={styles.app}>
            <aside className={styles.sidebar}><Sidebar /></aside>
            <main className={styles.main}><Outlet /></main>
            <footer className={styles.disclaimer}><Disclaimer /></footer>
        </div>
    )
}