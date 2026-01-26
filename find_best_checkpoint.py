import os
import json

# ==============================================================================
# SKRYPT DO ZNAJDOWANIA NAJLEPSZEGO CHECKPOINTU
# ==============================================================================

CHECKPOINT_DIR = "arterio_checkpoints"

def find_best_checkpoint():
    """
    Znajduje checkpoint z najniższym validation loss
    """
    print("=" * 80)
    print("🔍 SZUKAM NAJLEPSZEGO CHECKPOINTU")
    print("=" * 80)
    
    # Zbierz wszystkie checkpointy
    checkpoints = []
    
    for item in os.listdir(CHECKPOINT_DIR):
        item_path = os.path.join(CHECKPOINT_DIR, item)
        
        # Szukamy folderów checkpoint-XXX
        if os.path.isdir(item_path) and item.startswith("checkpoint-"):
            trainer_state_path = os.path.join(item_path, "trainer_state.json")
            
            if os.path.exists(trainer_state_path):
                # Wczytaj trainer_state.json
                with open(trainer_state_path, 'r') as f:
                    state = json.load(f)
                
                # Znajdź najlepszy validation loss dla tego checkpointu
                checkpoint_num = int(item.split("-")[1])
                
                # Szukamy w log_history
                eval_losses = []
                for entry in state.get("log_history", []):
                    if "eval_loss" in entry:
                        eval_losses.append({
                            "step": entry.get("step", checkpoint_num),
                            "epoch": entry.get("epoch", None),
                            "eval_loss": entry["eval_loss"]
                        })
                
                if eval_losses:
                    # Weź ostatni eval_loss dla tego checkpointu
                    last_eval = eval_losses[-1]
                    checkpoints.append({
                        "checkpoint": item,
                        "checkpoint_num": checkpoint_num,
                        "path": item_path,
                        "eval_loss": last_eval["eval_loss"],
                        "epoch": last_eval.get("epoch"),
                        "step": last_eval.get("step")
                    })
    
    if not checkpoints:
        print("❌ Nie znaleziono checkpointów z metrykami walidacyjnymi!")
        print(f"\n💡 Sprawdź czy w folderach checkpoint-XXX są pliki trainer_state.json")
        return None
    
    # Posortuj po eval_loss
    checkpoints.sort(key=lambda x: x["eval_loss"])
    
    print(f"\n📊 Znaleziono {len(checkpoints)} checkpointów:\n")
    
    # Wyświetl wszystkie checkpointy
    for i, cp in enumerate(checkpoints):
        marker = "🏆" if i == 0 else "  "
        epoch_str = f"Epoch {cp['epoch']:.2f}" if cp['epoch'] else f"Step {cp['step']}"
        print(f"{marker} {cp['checkpoint']:20s} | {epoch_str:15s} | Eval Loss: {cp['eval_loss']:.4f}")
    
    # Najlepszy checkpoint
    best = checkpoints[0]
    
    print("\n" + "=" * 80)
    print("🏆 NAJLEPSZY CHECKPOINT")
    print("=" * 80)
    print(f"Folder:     {best['checkpoint']}")
    print(f"Ścieżka:    {best['path']}")
    print(f"Eval Loss:  {best['eval_loss']:.4f}")
    if best['epoch']:
        print(f"Epoch:      {best['epoch']:.2f}")
    print(f"Step:       {best['step']}")
    
    # Sprawdź czy to jest ten sam co w głównym folderze
    main_adapter_path = os.path.join(CHECKPOINT_DIR, "adapter_model.safetensors")
    checkpoint_adapter_path = os.path.join(best['path'], "adapter_model.safetensors")
    
    print("\n" + "=" * 80)
    print("🔍 SPRAWDZAM KTÓRY MODEL JEST AKTUALNIE UŻYWANY")
    print("=" * 80)
    
    if os.path.exists(main_adapter_path):
        main_size = os.path.getsize(main_adapter_path)
        checkpoint_size = os.path.getsize(checkpoint_adapter_path)
        main_mtime = os.path.getmtime(main_adapter_path)
        checkpoint_mtime = os.path.getmtime(checkpoint_adapter_path)
        
        print(f"\n📁 Model w głównym folderze (arterio_checkpoints/):")
        print(f"   Rozmiar: {main_size:,} bytes")
        print(f"   Data:    {os.path.getmtime(main_adapter_path)}")
        
        print(f"\n📁 Model w najlepszym checkpoincie ({best['checkpoint']}):")
        print(f"   Rozmiar: {checkpoint_size:,} bytes")
        print(f"   Data:    {os.path.getmtime(checkpoint_adapter_path)}")
        
        if abs(main_mtime - checkpoint_mtime) < 1:
            print(f"\n✅ Model w głównym folderze JEST najlepszym checkpointem!")
            print(f"   Możesz używać: {CHECKPOINT_DIR}")
        else:
            print(f"\n⚠️  Model w głównym folderze NIE JEST najlepszym checkpointem!")
            print(f"   Powinieneś używać: {best['path']}")
    else:
        print(f"\n⚠️  Brak modelu w głównym folderze!")
        print(f"   Użyj checkpointu: {best['path']}")
    
    # Instrukcje
    print("\n" + "=" * 80)
    print("📝 CO DALEJ?")
    print("=" * 80)
    
    if os.path.exists(main_adapter_path):
        print(f"\n1️⃣ Jeśli chcesz używać najlepszego modelu, w evaluate_model.py ustaw:")
        print(f'   CHECKPOINT_DIR = "{best["path"]}"')
        print(f"\n   LUB sprawdź czy główny folder już go zawiera (patrz wyżej)")
    else:
        print(f"\n1️⃣ Użyj tego checkpointu do ewaluacji:")
        print(f'   CHECKPOINT_DIR = "{best["path"]}"')
    
    print(f"\n2️⃣ Porównaj eval_loss z innych checkpointów:")
    print(f"   - Jeśli rosną (np. {checkpoints[0]['eval_loss']:.3f} → {checkpoints[-1]['eval_loss']:.3f}) = OVERFITTING")
    print(f"   - Jeśli maleją = model się jeszcze uczył")
    
    print(f"\n3️⃣ Możesz też ręcznie sprawdzić plik:")
    print(f"   {os.path.join(best['path'], 'trainer_state.json')}")
    
    print("\n" + "=" * 80)
    
    return best

def check_training_progress():
    """
    Pokazuje jak zmieniał się loss podczas treningu
    """
    print("\n" + "=" * 80)
    print("📈 HISTORIA TRENINGU")
    print("=" * 80)
    
    trainer_state_path = os.path.join(CHECKPOINT_DIR, "trainer_state.json")
    
    if not os.path.exists(trainer_state_path):
        print("❌ Brak pliku trainer_state.json w głównym folderze")
        return
    
    with open(trainer_state_path, 'r') as f:
        state = json.load(f)
    
    log_history = state.get("log_history", [])
    
    print(f"\nŁącznie {len(log_history)} wpisów w historii\n")
    
    # Zbierz eval i train losses
    train_losses = []
    eval_losses = []
    
    for entry in log_history:
        if "loss" in entry and "eval_loss" not in entry:
            train_losses.append({
                "step": entry.get("step"),
                "epoch": entry.get("epoch"),
                "loss": entry["loss"]
            })
        if "eval_loss" in entry:
            eval_losses.append({
                "step": entry.get("step"),
                "epoch": entry.get("epoch"),
                "eval_loss": entry["eval_loss"]
            })
    
    print("🎯 TRAIN LOSS (ostatnie 10):")
    for entry in train_losses[-10:]:
        epoch_str = f"Epoch {entry['epoch']:.2f}" if entry['epoch'] else ""
        print(f"   Step {entry['step']:5d} {epoch_str:15s} | Loss: {entry['loss']:.4f}")
    
    print("\n📊 VALIDATION LOSS (wszystkie):")
    for entry in eval_losses:
        epoch_str = f"Epoch {entry['epoch']:.2f}" if entry['epoch'] else ""
        print(f"   Step {entry['step']:5d} {epoch_str:15s} | Eval Loss: {entry['eval_loss']:.4f}")
    
    # Analiza trendu
    if len(eval_losses) >= 2:
        first_eval = eval_losses[0]['eval_loss']
        last_eval = eval_losses[-1]['eval_loss']
        
        print("\n" + "=" * 80)
        print("🔍 ANALIZA TRENDU")
        print("=" * 80)
        print(f"Pierwszy eval loss:  {first_eval:.4f}")
        print(f"Ostatni eval loss:   {last_eval:.4f}")
        print(f"Zmiana:              {last_eval - first_eval:+.4f}")
        
        if last_eval < first_eval:
            print("✅ Model się poprawił podczas treningu!")
        else:
            print("⚠️  Validation loss wzrósł - możliwy overfitting!")
        
        # Sprawdź czy jest rosnący trend w ostatnich 3 ewaluacjach
        if len(eval_losses) >= 3:
            recent = [e['eval_loss'] for e in eval_losses[-3:]]
            if recent[0] < recent[1] < recent[2]:
                print("⚠️  Ostatnie 3 ewaluacje pokazują rosnący trend - OVERFITTING!")
            elif recent[0] > recent[1] > recent[2]:
                print("✅ Ostatnie 3 ewaluacje pokazują malejący trend - model się uczy!")

if __name__ == "__main__":
    best = find_best_checkpoint()
    check_training_progress()
    
    print("\n" + "=" * 80)
    print("✅ ANALIZA ZAKOŃCZONA")
    print("=" * 80)