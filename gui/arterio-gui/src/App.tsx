import {Sidebar} from "./components/Sidebar.tsx";
import {MainContent} from "./components/MainContent.tsx";
import {Disclaimer} from "./components/Disclaimer.tsx";
import styles from "./App.module.css";

function App() {
  return (
      <div className={styles.app}>
          <aside className={styles.sidebar}><Sidebar /></aside>
          <main className={styles.main}><MainContent /></main>
          <footer className={styles.disclaimer}><Disclaimer /></footer>
      </div>
  )
}

export default App
