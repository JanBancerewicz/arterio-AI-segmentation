import os
from tqdm import tqdm
from pipeline import InferencePipeline
from visualizer import ArterioVisualizer
from validator import Validator

CONFIG = {
    "base_model": "Qwen/Qwen2-VL-2B-Instruct",
    "adapter": "arterio_checkpoints",
    "input_dir": "../data/syntax/test/images",
    "output_dir": "final_results",
    "max_images": 3,  
    "prompt": "Detect and segment the coronary arteries. Return boxes."
}

def start_process():
    validator = new Validator()
  
    pipe = InferencePipeline(CONFIG["base_model"], CONFIG["adapter"])
    viz = ArterioVisualizer()
    
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    pipe.initialize_model()


    all_files = [f for f in os.listdir(CONFIG["input_dir"]) if f.lower().endswith(('.png', '.jpg'))]
    files_to_process = viz.natural_sort(all_files)
    
    if CONFIG["max_images"]:
        files_to_process = files_to_process[:CONFIG["max_images"]]

    print(f"[*] Rozpoczynam przetwarzanie {len(files_to_process)} zdjęć...")

 
    for filename in tqdm(files_to_process):
        img_path = os.path.join(CONFIG["input_dir"], filename)
        
        try:
            
            detected_boxes = pipe.run_inference(img_path, CONFIG["prompt"])

          
            result_img = viz.draw_bounding_boxes(img_path, detected_boxes)
            
      
            save_name = f"res_{filename}"
            result_img.save(os.path.join(CONFIG["output_dir"], save_name))

         
            is_valid = validator.validate_boxes(detected_boxes)
            
        except Exception as e:
            print(f"\n[Błąd] {filename}: {e}")

    print(f"\n[OK] Proces zakończony. Wyniki w: {CONFIG['output_dir']}")

if __name__ == "__main__":
    start_process()