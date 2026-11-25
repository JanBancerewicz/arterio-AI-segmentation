import {execFileSync} from 'node:child_process';
import {randomBytes} from 'node:crypto';
import path from 'node:path';

const projectRoot = path.resolve(import.meta.dirname, '../../..');

const compose = (args: string[], env: Record<string, string> = {}) =>
    execFileSync('docker', ['compose', ...args], {cwd: projectRoot, env: {...process.env, ...env}, encoding: 'utf8'});

export type Account = {username: string; password: string};

export const createAccount = (mustChangePassword: boolean): Account => {
    const account = {
        username: `e2e.${randomBytes(6).toString('hex')}`,
        password: randomBytes(18).toString('base64url'),
    };
    const script = 'import os, sys\nfrom app.users import create_user\n'
        + 'create_user(sys.argv[1], "Użytkownik E2E", os.environ["E2E_PASSWORD"], must_change_password=sys.argv[2] == "1")';
    compose(['exec', '-T', '-e', 'E2E_PASSWORD', 'api', 'python', '-c', script, account.username, mustChangePassword ? '1' : '0'],
        {E2E_PASSWORD: account.password});
    return account;
};

export const resetLoginLimits = () =>
    compose(['exec', '-T', 'db', 'psql', '-U', 'arterio', '-d', 'arterio', '-c', 'DELETE FROM login_limits']);
