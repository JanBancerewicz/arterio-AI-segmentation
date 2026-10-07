import {useParams} from "react-router";

export const AnalysisPage = () => {
    const { id } = useParams();
    return <>
        <h2>Analysis page for id: {id}</h2>
    </>
}