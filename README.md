# Kamerový systém s detekcí obličejů a červeného objektu

## Instalace

```bash
pip install -r requirements.txt
```

## Spuštění

```bash
python main.py
```

## Logika stavů

| Počet obličejů | Co se zobrazí |
|---|---|
| **0** | Obrazovka „Uživatel pryč" (tmavá + ikona osoby s křížem) |
| **1** | Live obraz + hledání červeného objektu s křížem |
| **2+** | Lock screen (zámek, zpráva „Zamčeno") |

## Detekce červeného objektu

- Převod do HSV barevného prostoru
- Dva rozsahy pro červenou (0–10° a 170–180° HUE)
- Morfologické filtrování šumu
- Kříž se vykreslí do středu největšího nalezeného červeného objektu

## Nastavení (v main.py)

| Proměnná | Význam | Výchozí |
|---|---|---|
| `MIN_RED_AREA` | Min. plocha červeného objektu (px²) | 500 |
| `CROSS_LEN` | Polovina délky ramene kříže | 30 |
| `CROSS_COLOR` | Barva kříže (BGR) | červená |

## Ovládání

- **Q** nebo **Escape** – ukončí program
