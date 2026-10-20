export type User = {
    id: string;
    username: string;
    displayName: string;
}

type LoginInput = {
    username: string;
    password: string;
}

export const login = async (input: LoginInput): Promise<User> => {
    await new Promise(resolve => setTimeout(resolve, 600));

    //mock for now
    if (
        input.username !== 'demo' ||
        input.password !== 'demo'
    ) {
        throw new Error('Nieprawidłowy identyfikator lub hasło.');
    }

    currentUser = {
        id: 'demouser',
        username: 'demo',
        displayName: 'Użytkownik Demonstracyjny',
    }

    return {...currentUser};
}

export const logout = async (): Promise<void> => {
    await new Promise(resolve => setTimeout(resolve, 300));

    currentUser = null;
};

let currentUser: User | null = null;

export const getSession = async (): Promise<User | null> => {
    return currentUser ? {...currentUser} : null;
};