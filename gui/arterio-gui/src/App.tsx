import {createBrowserRouter, createRoutesFromElements, Route} from "react-router";
import {AppLayout} from "./layouts/AppLayout.tsx";
import {NewAnalysisPage} from "./pages/NewAnalysisPage.tsx";
import {AnalysisPage} from "./pages/AnalysisPage.tsx";
import {AuthLayout} from "./layouts/AuthLayout.tsx";
import {LoginPage} from "./pages/LoginPage.tsx";
import {RequireSession} from "./components/RequireSession.tsx";

export const router = createBrowserRouter(createRoutesFromElements(
        <>
            <Route element={<RequireSession />} >
                <Route element={<AppLayout />} >
                    <Route index element={<NewAnalysisPage />} />
                    <Route path="analyses/:id" element={<AnalysisPage />} />
                </Route>
            </Route>
            <Route element={<AuthLayout />} >
                <Route path="/login" element={<LoginPage />} />
            </Route>
        </>
));
