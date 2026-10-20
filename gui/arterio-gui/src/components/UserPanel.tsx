import styles from './UserPanel.module.css';
import {useSession} from "../hooks/useSession.ts";
import {useLogout} from "../hooks/useLogout.ts";

const createAvatar = (displayName: string): string => {
    const splitedParts = displayName.trim().split(/\s+/).slice(0, 2);
    let avatar = "";
    for (const splitedPart of splitedParts) {
        avatar += splitedPart.charAt(0).toUpperCase();
    }
    return avatar;
}

export const UserPanel = () => {
    const {data: user} = useSession();
    const logoutMutation = useLogout();

    if(!user){
        return null;
    }

    return (
        <div className={styles.container}>
            <div className={styles.panel}>
                <div className={styles.avatar} aria-hidden="true">
                    {createAvatar(user.displayName)}
                </div>
                <div className={styles.info}>
                    <span className={styles.caption}>Zalogowany użytkownik</span>
                    <span className={styles.name} title={user.displayName}>
                        {user.displayName}
                    </span>
                </div>
            </div>
            <button
                    type="button"
                    className={styles.logout}
                    disabled={logoutMutation.isPending}
                    onClick={() => logoutMutation.mutate()}
                >
                    {logoutMutation.isPending ? 'Wylogowywanie...' : 'Wyloguj'}
                </button>
            {logoutMutation.isError && (
                <p className={styles.error} role="alert">
                    Nie udało się wylogować. Spróbuj ponownie.
                </p>
            )}
        </div>
    )
}
