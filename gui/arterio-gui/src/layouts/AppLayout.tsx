import { Outlet } from 'react-router'
import { Sidebar } from '../components/Sidebar.tsx'
import { Disclaimer } from '../components/Disclaimer.tsx'
import styles from './AppLayout.module.css'
import {useAnalysisEvents} from '../hooks/useAnalysisEvents.ts'
import {useState} from 'react'
import {MaskEditContext} from '../context/MaskEditContext.ts'

export const AppLayout = () => {
    useAnalysisEvents();
    const [dirty, setDirty] = useState(false);
    return (
        <MaskEditContext.Provider value={{dirty, setDirty}}>
        <div className={styles.app}>
            <aside className={styles.sidebar}><Sidebar /></aside>
            <main className={styles.main}><Outlet /></main>
            <footer className={styles.disclaimer}><Disclaimer /></footer>
        </div>
        </MaskEditContext.Provider>
    )
}
