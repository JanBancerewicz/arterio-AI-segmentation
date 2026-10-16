import styles from './ModelPicker.module.css';
import type {SegmentationModel} from "../../api/SegmentationModel.ts";

type ModelPickerProps = {
    segmentationModels: SegmentationModel[];
    selectedModelId: string | null;
    onSelect: (modelId: string) => void;
}

export const ModelPicker = (props: ModelPickerProps) => {
    return (
        <fieldset className={styles.picker}>
            <legend className={styles.legend}>Model</legend>

            <div className={styles.options}>
                {props.segmentationModels.map(model => (
                    <label key={model.id} className={styles.card}>
                        <input
                            type="radio"
                            name="model"
                            value={model.id}
                            checked={model.id === props.selectedModelId}
                            disabled={!model.available}
                            className={styles.radio}
                            onChange={() => props.onSelect(model.id)}
                        />
                        <span className={styles.body}>
                            <span className={styles.header}>
                                <span className={styles.name}>{model.name}</span>
                                <span className={styles.version}>v{model.version}</span>
                                {model.isDefault && <span className={styles.tag}>domyślny</span>}
                            </span>
                            <span className={styles.description}>{model.description}</span>
                        </span>
                    </label>
                ))}
            </div>
        </fieldset>
    )
}
