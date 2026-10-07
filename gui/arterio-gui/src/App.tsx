import {Route, Routes} from "react-router";
import {AppLayout} from "./layouts/AppLayout.tsx";
import {NewAnalysisPage} from "./pages/NewAnalysisPage.tsx";
import {AnalysisPage} from "./pages/AnalysisPage.tsx";
import {AuthLayout} from "./layouts/AuthLayout.tsx";
import {LoginPage} from "./pages/LoginPage.tsx";
import {RegisterPage} from "./pages/RegisterPage.tsx";

function App() {
    return (
        <Routes>
            <Route element={<AppLayout />} >
                <Route index element={<NewAnalysisPage />} />
                <Route path="analyses/:id" element={<AnalysisPage />} />
            </Route>
            <Route element={<AuthLayout />} >
                <Route path="/login" element={<LoginPage />} />
                <Route path="/register" element={<RegisterPage />} />
            </Route>
        </Routes>
    )
}

export default App
