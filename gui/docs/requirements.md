# Wymagania - warstwa prezentacji i backend aplikacyjny

## Kontekst
Model segmentacji naczyń wieńcowych udostępniany jest użytkownikowi w formie aplikacji webowej. Użytkownik może przesłać obraz angiograficzny w formacie PNG, który zostanie przetworzony asynchronicznie, a następnie zaprezentowany wraz z nałożoną maską naczyń oraz metadanymi analizy. Użytkownik ma także dostęp do historii swoich analiz.

<!-- TODO: dodać odniesienie do kontraktu API (endpointy, formaty danych, stany analizy, kody błędów) -->

## Słownik
| Pojęcie | Znaczenie |
|---------|-----------|
| Analiza | Jedno przesłane zdjęcie wraz z wybranym modelem, parametrami, statusem przetwarzania i (po zakończeniu) wynikiem. |
| Zadanie | Analiza oczekująca w kolejce lub będąca w trakcie przetwarzania. |
| Maska | Binarny obraz PNG tej samej rozdzielczości co obraz źródłowy: 255 = naczynie, 0 = tło. |
| Worker | Proces backendu, który pobiera zadania z kolejki i uruchamia model na GPU. |

## Aktorzy
- Użytkownik - lekarz; przesyła obrazy, przegląda wyniki
- Administrator - zarządza kontami (do zastanowienia się czy ta rola jest potrzebna)

## Wymagania funkcjonalne
| ID | Opis | Priorytet |
|----|------|-----------|
| WF-01 | Przesłanie obrazu PNG metodą drag-and-drop lub poprzez okno wyboru pliku | Must |
| WF-02 | Walidacja pliku: typ, rozmiar, rozdzielczość, ogólna poprawność <!-- TODO: odnośnik do reguł walidacji w kontrakcie API --> | Must |
| WF-03 | Utworzenie zadania do przetworzenia i umieszczenie go w kolejce | Must |
| WF-04 | Prezentacja statusu zadania w czasie rzeczywistym, w tym pozycji w kolejce i etapu przetwarzania | Must |
| WF-05 | Wyświetlenie wyniku: obraz źródłowy z nałożoną maską naczyń | Must |
| WF-06 | Wyświetlenie metadanych analizy: model i jego wersja, parametry, czas przetwarzania, znaczniki czasu, nazwa i rozdzielczość pliku | Must |
| WF-07 | Wybór modelu spośród dostępnych na liście (lub pozostawienie domyślnie wybranego) | Must |
| WF-08 | Przegląd historii wykonanych analiz z paginacją i filtrowaniem po statusie | Must |
| WF-09 | Ponowne otwarcie zapisanej w historii analizy | Must |
| WF-10 | Rejestracja i logowanie użytkownika | Must |
| WF-11 | Ograniczenie widoczności i dostępu do analiz wyłącznie do ich właściciela | Must |
| WF-12 | Wylogowanie użytkownika oraz automatyczne wygaśnięcie nieaktywnej sesji | Must |
| WF-13 | Obsługa i prezentacja błędów przetwarzania | Must |
| WF-14 | Anulowanie zadania oczekującego w kolejce | Must |
| WF-15 | Pobranie plików wynikowych (maska, obraz z nałożoną maską) | Should |
| WF-16 | Usunięcie analizy z historii wraz z powiązanymi plikami | Should |
| WF-17 | Ustawienie parametrów zaawansowanych: próg binaryzacji, TTA (o ile model je wspiera) | Could |
| WF-18 | Ponowienie analizy zakończonej błędem lub anulowanej, bez ponownego przesyłania pliku | Could |


## Wymagania interfejsu użytkownika
| ID | Opis | Priorytet |
|----|------|-----------|
| WI-01 | Prezentacja postępu przetwarzania bez konieczności odświeżania strony | Must |
| WI-02 | Czytelne komunikaty walidacji wskazujące przyczynę i sposób naprawy (np. „Plik ma 14 MB, maksymalnie 10 MB”) | Must |
| WI-03 | Widok porównawczy: obraz źródłowy obok wynikowego | Should |
| WI-04 | Włączanie/wyłączanie nakładki maski oraz regulacja jej przezroczystości | Should |
| WI-05 | Stała informacja, że narzędzie ma charakter badawczy i nie służy do diagnozy klinicznej | Must |
| WI-06 | Wstępna walidacja pliku po stronie przeglądarki (typ, rozmiar) przed wysłaniem | Should |
| WI-07 | Interfejs w języku polskim | Must |
| WI-08 | Układ dostosowany do ekranów desktopowych od 1280 px szerokości | Should |


## Wymagania niefunkcjonalne
| ID | Opis | Kryterium weryfikacji |
|----|------|-----------------------|
| WNF-01 | Komunikacja frontend-backend zgodna z kontraktem API | TODO: kryterium do uzupełnienia po przygotowaniu kontraktu API |
| WNF-02 | Maksymalny rozmiar przesyłanego pliku | Plik > 10 MB odrzucany z czytelnym komunikatem błędu |
| WNF-03 | Czas przetwarzania pojedynczego obrazu przy pustej kolejce | ≤ 30 s od przesłania do wyświetlenia wyniku (na docelowym GPU) |
| WNF-04 | Opóźnienie prezentacji zmiany statusu | Zmiana statusu widoczna w UI ≤ 2 s od jej wystąpienia w backendzie |
| WNF-05 | Trwałość kolejki | Restart backendu nie powoduje utraty analiz oczekujących w kolejce; analizy przerwane w trakcie przetwarzania kończą się błędem |
| WNF-06 | Bezpieczeństwo haseł | Hasła przechowywane wyłącznie jako hash (Argon2id lub bcrypt) |
| WNF-07 | Bezpieczeństwo sesji | Ciasteczko sesyjne `HttpOnly`, `SameSite=Lax`, w produkcji `Secure`; cała komunikacja produkcyjna po HTTPS |
| WNF-08 | Izolacja danych użytkowników | Żądanie o analizę lub plik innego użytkownika zwraca `404`; weryfikowane testem automatycznym |
| WNF-09 | Ochrona przed przeciążeniem | Maks. 5 aktywnych (oczekujących lub przetwarzanych) analiz na użytkownika; limit prób logowania |
| WNF-10 | Wspierane przeglądarki | Dwie ostatnie wersje Chrome, Firefox i Edge |
| WNF-11 | Odporność na zawieszenie przetwarzania | Analiza przetwarzana dłużej niż 120 s kończy się błędem przekroczenia czasu |


## Poza zakresem
- Obsługa formatu DICOM oraz sekwencji wideo (cine) - przyjmowane są wyłącznie pojedyncze klatki PNG
- Przesyłanie wielu plików naraz (batch upload)
- Ręczna edycja i korekta masek
- Segmentacja wieloklasowa (segmenty SYNTAX) oraz wykrywanie zwężeń
- Udostępnianie analiz innym użytkownikom
- Weryfikacja adresu e-mail i reset hasła drogą mailową
- Aplikacja mobilna
- Zastosowanie kliniczne - narzędzie nie jest wyrobem medycznym


## Założenia i ograniczenia
- Backend jest napisany w Pythonie (np. FastAPI), ponieważ modele są w PyTorch i można je importować bezpośrednio, bez osobnego serwisu.
- Frontend: React + TypeScript + Vite (`gui/arterio-gui`).
- Frontend i API są serwowane z jednego originu (w dev przez proxy Vite `/api` → backend, w produkcji przez reverse proxy) - dzięki temu nie jest potrzebny CORS, a ciasteczko sesyjne działa dla `<img>`, pobierania plików i SSE.
- Inferencja wymaga GPU z co najmniej 16 GB VRAM (Qwen3-VL-8B). Jeden worker przetwarza jedno zadanie naraz; kolejka jest FIFO.
- Modele zostały wytrenowane na zbiorze ARCADE (klatki 512×512). Obraz jest wewnętrznie skalowany do 512×512, a wynik wraca w rozdzielczości oryginału. Jakość dla obrazów spoza rozkładu ARCADE nie jest znana.
- Lista dostępnych modeli, ich wersje i model domyślny są konfigurowane po stronie backendu; frontend nie zna ich na sztywno.
- Pliki analizy są przechowywane do momentu jej usunięcia przez użytkownika (brak automatycznej retencji).


## Historia zmian
| Data | Zmiana |
|------|--------|
| 2026-09-13 | Wersja początkowa |
| 2026-10-02 | Poprawki redakcyjne; zastąpienie zduplikowanego WF-12 wymaganiem wylogowania/wygasania sesji; dodanie WF-17, WF-18, WI-04-WI-08, słownika, wymagań niefunkcjonalnych, zakresu i założeń |
