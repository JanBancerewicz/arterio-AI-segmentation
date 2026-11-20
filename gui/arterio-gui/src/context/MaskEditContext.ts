import {createContext, type Dispatch, type SetStateAction} from 'react';

export const MaskEditContext = createContext<{dirty: boolean; setDirty: Dispatch<SetStateAction<boolean>>}>({
    dirty: false, setDirty: () => {},
});
