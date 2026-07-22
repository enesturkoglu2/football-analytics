# Stage 5C jersey manual-review pilot sonuçları

## Amaç ve kapsam

Stage 5C-A jersey görünürlük/okunabilirlik pilotu,
`completed_manual_review_pilot_baseline` durumuyla tamamlandı. Pilot,
canonical 474 review item içinden deterministik seçilen 78 crop-level
item'ı kapsar. Seçim 42 critical item ve altı denge grubundan gelen 36
balance item'dan oluşur.

78 item'ın tamamı Furkan tarafından yedi manuel review partisinde
incelendi. Kalan 396 non-pilot satır unreviewed durumundadır; full 474
review yapılmadı.

## Kaynak artifact ve provenance

Kaynak annotation:
`outputs/reid/full_stage4b/jersey_review_pilot_stage5c/pilot_annotations_working.csv`

- SHA-256:
  `6cf90d3a6fa94025020dac8b1bc63d5ca51105589bdbbfeb8007d45356cb7b63`
- Canonical row: 474
- Reviewed: 78
- Unreviewed: 396
- Distinct reviewer: 1
- Review batch: 7

Kaynak template, selection, guide, summary, review artifact'leri,
manual-review config, 83 panel PNG ve 474 source crop dahil 567 kaynak
dosya freeze öncesi ve sonrası SHA-256, byte size ve `mtime_ns` ile
karşılaştırıldı. Değişen kaynak sayısı sıfırdır.

## Final dağılım

| Alan | Dağılım |
|---|---|
| `manual_crop_valid` | valid=65, uncertain=5, invalid=8 |
| `manual_back_facing` | no=35, yes=27, uncertain=8, blank=8 |
| `manual_number_visible` | no=37, yes=26, uncertain=7, blank=8 |
| `manual_number_readable` | no=41, yes=20, uncertain=9, blank=8 |
| `manual_digit_count` | 0=37, 1=9, 2=13, uncertain=11, blank=8 |
| `manual_contamination_affects_number_region` | no=57, yes=10, uncertain=3, blank=8 |

Dinamik oranlar:

| Oran | Pay / payda | Değer |
|---|---:|---:|
| Valid crop / pilot | 65 / 78 | 0.833333 |
| Invalid crop / pilot | 8 / 78 | 0.102564 |
| Visible yes / pilot | 26 / 78 | 0.333333 |
| Readable yes / pilot | 20 / 78 | 0.256410 |
| Readable yes / valid crop | 20 / 65 | 0.307692 |
| Readable yes / visible-yes crop | 20 / 26 | 0.769231 |

## Okunabilir jersey observation dağılımı

Toplam 20 crop-level readable jersey observation vardır:

| Jersey | Observation |
|---:|---:|
| 9 | 4 |
| 17 | 5 |
| 10 | 2 |
| 3 | 1 |
| 8 | 1 |
| 30 | 3 |
| 7 | 2 |
| 19 | 1 |
| 2 | 1 |

## Segment ve purity bulguları

- `raw_231_s01`: frame 268 ve 277 üzerinde readable jersey 9.
- `raw_231_s02`: frame 290, 298, 339, 350 ve 359 üzerinde readable
  jersey 17.
- `raw_514_s01`: frame 446 jersey 3, frame 454 jersey 8.
  `purity_warning=true`.
- `raw_514_s02`: frame 463, 479 ve 496 jersey 30; frame 487 visible
  fakat unreadable.
- `raw_16_full`: beş invalid/off-pitch crop.
  `pass_through_purity_warning=true`.
- `raw_13_full`: üç invalid/off-pitch crop.
  `pass_through_purity_warning=true`.
- `raw_738_full`: mixed-player crop; görünür 17 target'a atanmadı.
  `mixed_target_warning=true`.
- `raw_639_full`: number-like/two-digit observation'lar uncertain;
  jersey atanmadı.
- `raw_806_full`: visible fakat uncertain/unreadable observation'lar.
- `raw_558_full`: visible marking mevcut, okuma uncertain.

Bu bulgular mevcut insan onaylı CSV değerleri ve notlarından
doğrulanmıştır; yeni görsel yorum veya sayı üretilmemiştir.

## Validator sonucu

Existing validator `status=valid` ve exit code 0 döndürdü:

- expected/actual row: 474/474
- reviewed/unreviewed: 78/396
- error: 0
- warning: 0
- duplicate/missing/extra/provenance mismatch: 0
- bütün safety alanları: false

## Freeze artifact'leri

Freeze dizini:
`outputs/reid/full_stage4b/jersey_pilot_results_stage5c`

| Artifact | SHA-256 | Byte | Row |
|---|---|---:|---:|
| `pilot_annotations_frozen.csv` | `6cf90d3a6fa94025020dac8b1bc63d5ca51105589bdbbfeb8007d45356cb7b63` | 101450 | 474 |
| `pilot_validation_report.json` | `83ba0e551763f4db64ed3b6768f49b615497c136118d067173440834931466f1` | 3031 | 1 |
| `pilot_reviewed_items.jsonl` | `1ff03783649d3890f748b12c44d798731fdc41d96c6a005a9f4dc3b0e93c4eca` | 95331 | 78 |
| `pilot_results_summary.json` | `795e027376ba2e19af89615c53de8a106773475893fa63a83e6343d52fcbce1b` | 4862 | 1 |
| `pilot_freeze_manifest.json` | `19d848dea4da474c74716f1bf19476238eacbf7d68cc3e90771fe064042150aa` | 4394 | 1 |

Frozen CSV working CSV ile byte-identical'dır. Freeze paketi yalnız
yukarıdaki beş artifact'i içerir; medya, model veya checkpoint içermez.

## Interpretation limits

- Bunlar single-reviewer, Furkan-approved crop-level annotation'lardır.
- Independent double-review yapılmadı.
- Sonuçlar physical identity ground truth değildir.
- Jersey number tek başına physical identity değildir.
- Selected crop'lar bütün observation'lar değildir.
- Pilot, 474 crop'u temsil eden bir accuracy benchmark değildir.
- No-crop veya invalid crop negatif identity kanıtı değildir.
- Readable observation bir model accuracy sonucu değildir.
- Pilot sonucu checkpoint performance sonucu değildir.
- `unknown` ve `uncertain` durumları korunmuştur.
- OCR/recognizer/checkpoint veya model inference kullanılmadı.
- Identity, team, global-ID, component, link veya gallery güncellenmedi.

## Stage 5C-B çıkış kriteri

Stage 5C-A manual review tamamlanmıştır. Sonraki kapı **Stage 5C-B
recognizer/checkpoint capability audit**'tir. Bu audit, mevcut freeze
paketini immutable crop-level baseline olarak kullanmalı; checkpoint
varlığını ve recognizer capability'sini ayrı doğrulamalı, bu pilotu
accuracy veya identity ground truth olarak yorumlamamalıdır.
