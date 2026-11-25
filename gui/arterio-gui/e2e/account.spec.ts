import {expect, type Page, test} from '@playwright/test';
import {createAccount} from './stack.ts';

const login = async (page: Page, username: string, password: string) => {
    await page.goto('/login');
    await page.getByLabel('Identyfikator').fill(username);
    await page.getByLabel('Hasło', {exact: true}).fill(password);
    await page.getByRole('button', {name: 'Zaloguj się'}).click();
};

test('anonymous visitor is sent to the login page', async ({page}) => {
    await page.goto('/');
    await expect(page).toHaveURL(/\/login$/);
    await expect(page.getByRole('heading', {name: 'Zaloguj się'})).toBeVisible();
});

test('invalid credentials are rejected', async ({page}) => {
    await login(page, 'nie.istnieje', 'zle-haslo-123456');
    await expect(page.getByRole('alert')).toHaveText('Nieprawidłowy identyfikator lub hasło.');
    await expect(page).toHaveURL(/\/login$/);
});

test('first login requires replacing the temporary password', async ({page}) => {
    const account = createAccount(true);
    const newPassword = `${account.password}-nowe`;

    await login(page, account.username, account.password);
    await expect(page).toHaveURL(/\/change-password$/);
    await expect(page.getByRole('heading', {name: 'Ustaw własne hasło'})).toBeVisible();

    await page.goto('/');
    await expect(page).toHaveURL(/\/change-password$/);

    await page.getByLabel('Hasło tymczasowe').fill(account.password);
    await page.getByLabel('Nowe hasło', {exact: true}).fill(newPassword);
    await page.getByLabel('Powtórz nowe hasło').fill(newPassword);
    await page.getByRole('button', {name: 'Zmień hasło'}).click();
    await expect(page.getByRole('heading', {name: 'Nowa analiza'})).toBeVisible();

    await page.getByRole('button', {name: 'Wyloguj'}).click();
    await expect(page).toHaveURL(/\/login$/);

    await login(page, account.username, account.password);
    await expect(page.getByRole('alert')).toHaveText('Nieprawidłowy identyfikator lub hasło.');

    await login(page, account.username, newPassword);
    await expect(page.getByRole('heading', {name: 'Nowa analiza'})).toBeVisible();
    await expect(page.getByText('Użytkownik E2E')).toBeVisible();
});

test('session survives a page reload and ends after logout', async ({page}) => {
    const account = createAccount(false);
    await login(page, account.username, account.password);
    await expect(page.getByRole('heading', {name: 'Nowa analiza'})).toBeVisible();

    await page.reload();
    await expect(page.getByRole('heading', {name: 'Nowa analiza'})).toBeVisible();

    await page.getByRole('button', {name: 'Wyloguj'}).click();
    await expect(page).toHaveURL(/\/login$/);
    await page.goto('/');
    await expect(page).toHaveURL(/\/login$/);
});
