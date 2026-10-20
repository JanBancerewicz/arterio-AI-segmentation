import styles from './Title.module.css';

type TitleProps = {
    variant?: 'light' | 'dark';
};

export const Title = ({variant = 'light'}: TitleProps) => {
    return (
        <div className={`${styles.title} ${styles[variant]}`}>
            <h2 className={styles.name}>ARTERIO</h2>
        </div>
    );
};
