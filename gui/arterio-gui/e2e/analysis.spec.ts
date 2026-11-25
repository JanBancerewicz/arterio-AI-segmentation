import {expect, type Page, test} from '@playwright/test';
import {createAccount} from './stack.ts';

const angiographyFrame = async (page: Page) => {
    const dataUrl = await page.evaluate(() => {
        const canvas = document.createElement('canvas');
        canvas.width = 512;
        canvas.height = 512;
        const context = canvas.getContext('2d')!;
        context.fillStyle = '#202020';
        context.fillRect(0, 0, 512, 512);
        context.strokeStyle = '#f0f0f0';
        context.lineWidth = 14;
        context.beginPath();
        context.moveTo(40, 80);
        context.bezierCurveTo(200, 160, 300, 300, 470, 440);
        context.stroke();
        return canvas.toDataURL('image/png');
    });
    return Buffer.from(dataUrl.split(',')[1], 'base64');
};

const signIn = async (page: Page) => {
    const account = createAccount(false);
    await page.goto('/login');
    await page.getByLabel('Identyfikator').fill(account.username);
    await page.getByLabel('Hasło', {exact: true}).fill(account.password);
    await page.getByRole('button', {name: 'Zaloguj się'}).click();
    await expect(page.getByRole('heading', {name: 'Nowa analiza'})).toBeVisible();
};

const startAnalysis = async (page: Page, filename: string) => {
    await page.locator('input[type="file"]').setInputFiles({
        name: filename, mimeType: 'image/png', buffer: await angiographyFrame(page),
    });
    await expect(page.getByAltText(`Podgląd pliku ${filename}`)).toBeVisible();
    await page.getByRole('button', {name: 'Rozpocznij analizę'}).click();
    await expect(page).toHaveURL(/\/analyses\/[0-9a-f-]{36}$/);
    await expect(page.getByText('Analiza została zakończona.')).toBeVisible({timeout: 30_000});
};

test('user uploads a frame, receives the segmentation and finds it in history', async ({page}) => {
    await signIn(page);
    await startAnalysis(page, 'klatka-e2e.png');

    const results = page.getByRole('group', {name: 'Wybór obrazu'});
    for (const label of ['Oryginał', 'Maska', 'Nałożenie maski']) {
        await results.getByRole('button', {name: label}).click();
        await expect(results.getByRole('button', {name: label})).toHaveAttribute('aria-pressed', 'true');
        await expect(page.locator('section[aria-labelledby="analysis-image-heading"] img')).toBeVisible();
    }
    await expect(page.getByText('512 × 512 px')).toBeVisible();

    const historyEntry = page.getByRole('link', {name: /klatka-e2e\.png/});
    await expect(historyEntry).toBeVisible();
    await expect(historyEntry).toContainText('Zakończona');

    await page.getByRole('link', {name: 'Przejdź do nowej analizy'}).click();
    await historyEntry.click();
    await expect(page.getByText('Analiza została zakończona.')).toBeVisible();
});

test('user corrects the mask, downloads it and the guard protects unsaved edits', async ({page}) => {
    await signIn(page);
    await startAnalysis(page, 'edycja-maski.png');

    await page.getByRole('button', {name: 'Edytuj maskę'}).click();
    const canvas = page.getByLabel('Obraz z edytowaną maską');
    await expect(canvas).toBeVisible();
    await expect(page.getByText('Maska zgodna z predykcją.')).toBeVisible();

    const box = (await canvas.boundingBox())!;
    await page.mouse.move(box.x + box.width * 0.2, box.y + box.height * 0.7);
    await page.mouse.down();
    await page.mouse.move(box.x + box.width * 0.6, box.y + box.height * 0.75, {steps: 10});
    await page.mouse.up();
    await expect(page.getByText('Masz lokalne poprawki.')).toBeVisible();

    await page.getByRole('button', {name: 'Cofnij'}).click();
    await expect(page.getByText('Maska zgodna z predykcją.')).toBeVisible();
    await page.getByRole('button', {name: 'Ponów'}).click();
    await expect(page.getByText('Masz lokalne poprawki.')).toBeVisible();

    const download = page.waitForEvent('download');
    await page.getByRole('button', {name: 'Pobierz poprawioną maskę PNG'}).click();
    expect((await download).suggestedFilename()).toMatch(/\.png$/);

    await expect(page.getByRole('button', {name: 'Wyloguj'})).toBeDisabled();
    await page.getByRole('button', {name: 'Zakończ edycję'}).click();
    await expect(page.getByRole('alertdialog', {name: 'Odrzucić lokalne poprawki?'})).toBeVisible();
    await page.getByRole('button', {name: 'Kontynuuj edycję'}).click();
    await expect(page.getByText('Masz lokalne poprawki.')).toBeVisible();

    await page.getByRole('button', {name: 'Zakończ edycję'}).click();
    await page.getByRole('button', {name: 'Odrzuć poprawki i wyjdź'}).click();
    await expect(page.getByRole('button', {name: 'Edytuj maskę'})).toBeVisible();
});

test('user deletes a finished analysis', async ({page}) => {
    await signIn(page);
    await startAnalysis(page, 'do-usuniecia.png');

    await page.getByRole('button', {name: 'Usuń analizę'}).click();
    await page.getByRole('button', {name: 'Zachowaj analizę'}).click();
    await page.getByRole('button', {name: 'Usuń analizę'}).click();
    await page.getByRole('button', {name: 'Usuń analizę i pliki'}).click();

    await expect(page.getByRole('heading', {name: 'Nowa analiza'})).toBeVisible();
    await expect(page.getByRole('link', {name: /do-usuniecia\.png/})).toHaveCount(0);
});

test('invalid file is rejected before upload', async ({page}) => {
    await signIn(page);
    await page.locator('input[type="file"]').setInputFiles({
        name: 'notatki.png', mimeType: 'image/png', buffer: Buffer.from('to nie jest obraz'),
    });
    await expect(page.getByText(/zawartość nie jest obrazem PNG/)).toBeVisible();
    await expect(page.getByRole('button', {name: 'Rozpocznij analizę'})).toBeDisabled();
});

test('users cannot see each other analyses', async ({browser}) => {
    const owner = await browser.newPage();
    await signIn(owner);
    await startAnalysis(owner, 'prywatna.png');
    const privateUrl = owner.url();

    const stranger = await (await browser.newContext()).newPage();
    await signIn(stranger);
    await expect(stranger.getByRole('link', {name: /prywatna\.png/})).toHaveCount(0);
    await stranger.goto(privateUrl);
    await expect(stranger.getByRole('heading', {name: 'Nie znaleziono analizy'})).toBeVisible();
});
