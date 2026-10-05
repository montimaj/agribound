# International Landsat PAN/SR comparison

Example 25 (`examples/25_landsat_international_comparison.py`)
extends the [existing comparison](landsat-pan-comparison.md) to six fixed areas
in four countries outside France. On October 1, 2026, five areas completed
all three experiments: **15 actual new delineations**, using `ee-rappjer` and
the existing pinned `large_v2` checkpoint. The Red River Delta area remains
unmeasured because no scenes meet the fixed selection. French outputs were
read for comparison; their inference was not repeated.

## Reproduce the suite

Run from the repository root with Python >=3.12:

```bash
python -m pip install -e ".[gee,delineate-anything]" matplotlib
agribound auth --project ee-rappjer
python examples/25_landsat_international_comparison.py --gee-project ee-rappjer
python examples/25_landsat_international_comparison.py --summarize-only
```

For the versions used in the live Windows/Python 3.12 run, install with
`-c examples/landsat_comparison_constraints.txt`. These are tested environment
pins, not a claim that every version has wheels for every operating system.
Use the constraint file with the same installation command:

```bash
python -m pip install -c examples/landsat_comparison_constraints.txt \
  -e ".[gee,delineate-anything]" matplotlib
```

The model revision and SHA-256 are the same as example 23. Authentication
requires an Earth Engine-enabled Google account and a registered project with
access to the public Landsat collections. Reference downloads need no account.

The executed command reused the local checkpoint:

```bash
python examples/25_landsat_international_comparison.py --gee-project ee-rappjer \
  --checkpoint outputs/landsat_pan_sr_comparison/model_cache/hub/models--MykolaL--DelineateAnything/snapshots/369d0b4c44cf9bec2bd3a27bc81810cadd2c963e/DelineateAnythingv2.pt
```

`--sites nl_meierij es_olite` runs a subset. `--prepare-only` prepares inputs
and references; `--offline --checkpoint PATH` uses prepared inputs without
Earth Engine. Completed sites are reused, with configuration checks. Use a
new `--output-dir` for changed selection criteria. A suite run exits nonzero
when any requested site remains unmeasured, while retaining completed outputs.
`--summarize-only` succeeds without imagery access or model inference.

On Windows, if another application's environment points `PROJ_LIB`, `PROJ_DATA`
or `GDAL_DATA` at an incompatible installation, clear those variables in the
current shell before running. The geospatial wheels then use their own data:

```powershell
Remove-Item Env:PROJ_LIB,Env:PROJ_DATA,Env:GDAL_DATA -ErrorAction SilentlyContinue
```

## Frozen selection and reference inventory

The frozen site configuration (`examples/landsat_international_sites.json`)
records boxes, dates, evidence, selection rationale, source URLs, licenses and
reference-file SHA-256 pins. `sites_frozen.json` is written before inference;
changed site selection requires a different output folder. Selection used
reference geometry and reported agricultural context, without model scores.
Boxes are deliberately small, and Vietnamese boxes are inside digitized tiles.

| Area and administrative region | Regional climate / cultivation context | Inclusive imagery window | Reference count; median area |
| --- | --- | --- | --- |
| Meierij, Noord-Brabant, Netherlands | Temperate maritime; maize, potatoes and sugar beet declarations | May–August 2022 | 280; 2.07 ha |
| Valtierra, Ribera del Ebro, Navarra, Spain | Dry Mediterranean continental / semi-arid regional context; arable and horticultural use, documented regional irrigation | May–August 2020 | 369; 0.75 ha |
| Olite, Zona Media, Navarra, Spain | Mediterranean dry summers; arable, vineyard and olive enclosures | April–July 2020 | 622; 0.49 ha |
| Heidelberg/Hessequa, Western Cape, South Africa | Southern Cape winter-growing grain/forage region; larger parcels | July–October 2018 | 87; 19.16 ha |
| Thai Binh, Red River Delta, Vietnam | Northern monsoonal climate; rice-based smallholder regional context | February–May 2021 | 1,214; 0.29 ha; unmeasured |
| Dong Thap, Mekong Delta near Cao Lanh, Vietnam | Humid tropical monsoonal; dry-season, multiple-rice-cropping regional context | January–April 2021 | 461; 1.24 ha |

Vietnamese administrative names refer to 2021. Crop species are verified from
provider attributes where present, not from image appearance. Netherlands
declarations include 175 silage-maize and 30 consumption-potato parcels in the
selected box. Olite includes 116 vineyard and 114 olive enclosures; its 350
ARABLE LAND labels identify a use class, not a crop species. Hessequa includes
29 wheat, seven barley, five canola, four small-grain grazing and 42
lucerne/medics labels. Vietnamese polygons have **no species attribute**;
rice is regional context and cannot support field-level crop stratification.
Likewise, Valtierra's irrigation context does not verify every field as irrigated.

Climate evidence comes from
[KNMI](https://cdn.knmi.nl/knmi/pdf/bibliotheek/knmipubmetnummer/knmipub102-87.pdf),
[Meteo Navarra's southern climate description, including Olite and the Ribera](https://meteo.navarra.es/climatologia/zona_sur.cfm),
and the [World Bank/ADB Vietnam climate profile](https://openknowledge.worldbank.org/server/api/core/bitstreams/cd098629-b2fa-50d9-b4c2-585a55010cb5/content).
[Hessequa's government profiles](https://www.westerncape.gov.za/treasury/socio-economic-profiles-2022)
provide regional socioeconomic context, rather than a formal climate classification.
The original frozen 2015 profile URL is retained in research provenance but now
returns 404; this current provider index does not change the frozen experiments.
[FAO's rice calendars](https://www.fao.org/fileadmin/templates/rap/files/meetings/2016/161109_AMIS_4.7-AMIS_Stock_Seminar_Vietnam_rev.pdf)
support different northern and southern Vietnamese windows;
[FAO's cultivation description](https://www.fao.org/4/Y4347E/y4347e1u.htm)
supports rice-based systems in both deltas. South African July–October is a
Southern Hemisphere winter/spring window, consistent with the
[ESA challenge's April–November winter-crop season](https://platform-challenges.philab.esa.int/ai4food-security-south-africa).
Navarra documents [Valtierra irrigation](https://www.lexnavarra.navarra.es/detalle.asp?r=29769)
and [agricultural use categories in the 2023 SIGPAC annex](https://www.lexnavarra.navarra.es/detalle.asp?r=56175).
These maintained provider pages replace unavailable documentation links; the
original evidence URLs, 2020 SIGPAC attributes and frozen experiments remain
unchanged in research provenance. The newer SIGPAC annex explains use codes,
not the crop or geometry vintage of the evaluated polygons.
These contextual sources do not establish causal differences in model performance.

| Reference inventory | License and origin | Interpretation and limits |
| --- | --- | --- |
| [Netherlands BRP 2022 subset](https://source.coop/kerner-lab/fields-of-the-world-netherlands) | CC0-1.0; RVO/PDOK, converted by Kerner Lab | Annual crop declarations based on an agricultural register. This subset includes arable classes and omits grassland; it is not exhaustive physical-field truth. [PDOK describes the declarations](https://www.pdok.nl/ogc-apis/-/article/gewaspercelen-inspire-geharmoniseerd-). |
| [Spain/Navarra 2020 subset](https://source.coop/kerner-lab/fields-of-the-world-spain) | CC-BY-4.0; SIGPAC/EuroCrops, converted by Kerner Lab | Agricultural land-use enclosures; selected classes only. Ownership, use and visible boundaries need not coincide. |
| [South Africa nominal 2018 subset](https://source.coop/kerner-lab/fields-of-the-world-southafrica) | CC-BY-NC-SA-4.0 in the conversion; [original provider document](https://radiantearth.blob.core.windows.net/mlhub/esa-food-security-challenge/Crops_GT_Western_Cape_Doc.pdf) also restricts distribution/commercial use and states competition/academic research scope | Crop-labeled reference, five classes. Provider geometry mainly uses late-2016 aerial images, digitized in 2017, with exceptional January-2018 edits; crop surveys ran May 2017–March 2018. July–October 2018 contemporaneity is unverified. |
| [Vietnam 2021 AI4SmallFarms](https://source.coop/kerner-lab/fields-of-the-world-vietnam) | CC-BY-4.0; [original dataset](https://doi.org/10.17026/dans-xy6-ngg6), converted by Kerner Lab | Manual digitization of all visible fields within surveyed tiles, with quality/topology checks. August 2021 reference imagery is later than the tested seasonal windows, even though the year matches. |

None of these conversion records supplies a numerical positional-error bound.
Digitization resolution and scale do not establish positional accuracy. The
South African provenance explicitly records the original dates and marks
`vintage_matches=false`; this documentation correction changed neither
reference geometry nor measured metrics. Extracted references stay in ignored
outputs; South African vectors and its reference-overlay map are not copied
into release assets because of the additional provider restrictions.

Presence-only Indian, Kenyan and Rwandan FTW references were unsuitable for
an unrestricted whole-box precision comparison. Machine-generated global
boundary maps were excluded as independent truth. The
[FTW dataset description](https://github.com/fieldsoftheworld/ftw-datasets-list)
and [TorchGeo's FTW label documentation](https://docs.torchgeo.org/en/latest/api/datasets/fields-of-the-world.html)
and [AI4SmallFarms paper](https://elib.dlr.de/200095/1/AI4SmallFarms_A_Dataset_for_Crop_Field_Delineation_in_Southeast_Asian_Smallholder_Farms.pdf)
document relevant coverage distinctions. No guaranteed geographical holdout
from the released Delineate-Anything checkpoint is established.

## Measured results and the unmeasured site

| Area | Detection F1: PAN / SR / combined | Boundary F1 at 15 m: PAN / SR / combined | CPU inference seconds: PAN / SR / combined |
| --- | --- | --- | --- |
| Meierij | 0.254 / 0.134 / 0.243 | 0.452 / 0.262 / 0.424 | 25.0 / 7.2 / 17.1 |
| Valtierra | 0.135 / 0.038 / 0.098 | 0.505 / 0.321 / 0.441 | 18.9 / 6.9 / 13.8 |
| Olite | 0.210 / 0.056 / 0.178 | 0.560 / 0.273 / 0.508 | 21.5 / 6.5 / 15.9 |
| Hessequa, vintage caveat above | 0.415 / 0.466 / 0.547 | 0.498 / 0.323 / 0.504 | 17.6 / 6.9 / 13.1 |
| Red River Delta | **Unmeasured** | **Unmeasured** | **Unmeasured** |
| Mekong Delta | 0.000 / 0.000 / 0.000 | 0.218 / 0.326 / 0.325 | 12.2 / 6.1 / 11.2 |

Detection uses one-to-one IoU >=0.5. PAN and combined outputs have 15 m
input grids; SR has 30 m. Timings are single CPU runs, exclude download and
shared preparation, and include different pixel-based tile/context sizes.
The model was trained at 0.25–10 m; both input resolutions are outside that
range. The RGB engine uses visible SR bands; this does not evaluate a model
specifically designed to exploit NIR/SWIR.

Matched scene counts are 7, 3, 6, 11 and 1 respectively for completed areas,
with zero unmatched eligible scenes. The Mekong composite has 96.63% valid
export-grid support, compared with 100% at the other completed sites. Support
and observations are identical among all three experiments within each site.
Mekong therefore represents one acquisition, not a multi-scene seasonal median.
All references remain in the evaluation, including areas without valid imagery;
this coverage penalty is shared across inputs and disclosed in provenance.

The Red River Delta selection has **zero eligible scenes** in both products
under the fixed 20% scene-cloud threshold from February 1 through May 31, 2021.
`inputs/scene_selection_failure.json` preserves dates, selection, empty pairs
and unmatched lists. No imagery, prediction GeoPackages or accuracy metrics
exist for that site. Its reference extract and configuration are complete.
The threshold/window and study box were retained after this failure.

Size-stratified detection F1 illustrates the limits hidden by overall results:

| Site; size class | Reference fields | PAN | SR | Combined |
| --- | ---: | ---: | ---: | ---: |
| Meierij; 1–5 ha | 157 | 0.249 | 0.073 | 0.267 |
| Meierij; 5–20 ha | 40 | 0.583 | 0.453 | 0.506 |
| Olite; <1 ha | 424 | 0.004 | 0.000 | 0.000 |
| Olite; 5–20 ha | 30 | 0.706 | 0.536 | 0.678 |
| Hessequa; 20–100 ha | 43 | 0.422 | 0.682 | 0.624 |
| Mekong; 1–5 ha | 275 | 0.000 | 0.000 | 0.000 |

Full size bins, 10/15/30 m boundary precision/recall/F1, matched polygon IoU,
over/undersegmentation and per-field matches are retained in CSV/JSON.
Empty size classes remain undefined, not zero. Many Mekong reference parcels
are elongated strips: median rectangle elongation is 7.57. All three runs
merge or miss most internal boundaries; a nonzero boundary F1 does not mean
individual fields were detected. A larger area in hectares does not guarantee
a sufficiently wide field for the sensor/model.

## Aggregation and interpretation

These summaries include the five measured international sites, 1,819 references,
and exclude the unmeasured northern Vietnamese site:

| Weighting | Detection F1: PAN / SR / combined | Boundary F1 at 15 m: PAN / SR / combined |
| --- | --- | --- |
| Equal site | 0.203 / 0.139 / 0.213 | 0.447 / 0.301 / 0.440 |
| Reference count | 0.158 / 0.070 / 0.144 | 0.442 / 0.297 / 0.435 |

These are weighted means of site metrics, **not pooled F1 or pooled boundary
length scores**. The field-count weighting emphasizes dense small-parcel
sites, while equal-site weighting gives Hessequa's 87 parcels as much influence
as Olite's 622. Undefined metrics are excluded with per-metric contributing
site counts. Neither summary removes differences in reference semantics,
seasons, acquisition counts, positional uncertainty or possible model overlap.
Country, crop and climate effects are not causally identified.

PAN improves boundary F1 over SR in Meierij and both Navarra boxes, extending
the four French cases. SR is better for the Mekong boundary measure and
Hessequa's largest-field detection class. Fusion improves Hessequa's overall
detection F1 over both standalone inputs; its boundary gain over PAN is only
0.006 and comes with the older-reference caveat. Fusion loses to PAN in the
three European boxes and is effectively tied with SR in Mekong while all
field-detection scores there remain zero. Its complexity is therefore a
site-dependent tradeoff, not a generally demonstrated improvement.

The fusion remains **image-level visible RGB detail injection**. PAN is TOA
reflectance and the coarse RGB values are SR. Resampling SR adds no spatial
detail; only PAN supplies the injected residual. Each site has coarse-mean
preservation and reduced-resolution diagnostics. These do not establish
calibrated 15 m SR: box degradation is not MTF matching. In Hessequa, red/green
reconstruction RMSE improves but blue RMSE and median spectral angle worsen
(0.687 degrees resampled versus 0.947 degrees fused). Better delineation does
not establish better spectral fidelity.

## Artifacts and validation

`outputs/landsat_international/<site>/` contains reference GeoPackages,
provider metadata, paired inputs/manifests, fusion diagnostics, three predicted
GeoPackages with provenance, all metrics, run logs and map PNG/PDFs. Every
map uses the same 30 m SR background in all columns, identical extents and
reference-selected small-field/shared-edge views. The blocked site contains
its reference and failure evidence only.

Top-level `international_comparison.csv` combines all measured international
and existing French strata; `headline_15m.csv` and `aggregate_metrics.csv`
provide smaller summaries. `suite_status.json` reports per-site completion,
vintage, scene counts, coverage and parcel profiles. The cross-site figure
includes the French results and explicitly identifies the unmeasured site.

Measured international and French comparison (`outputs/landsat_international/international_comparison.png`; generated locally)

Recorded metrics (`outputs/landsat_international/international_comparison.csv`; generated locally),
aggregates (`outputs/landsat_international/aggregate_metrics.csv`; generated locally),
status/source inventory (`outputs/landsat_international/suite_status.json`; generated locally)
and maps for Meierij (`outputs/landsat_international/nl_meierij.png`; generated locally),
Valtierra (`outputs/landsat_international/es_valtierra.png`; generated locally),
Olite (`outputs/landsat_international/es_olite.png`; generated locally)
and Mekong (`outputs/landsat_international/vn_mekong.png`; generated locally)
accompany the guide. Hessequa's generated map remains in its local output folder.

Meaningful offline tests cover pinned/atomic downloads, corrupt-cache rejection,
CRS-aware whole-polygon selection, invalid geometry repair, duplicate IDs,
reference vintage, source/crop provenance, frozen selection, weighting,
documentation-only corrections and failure evidence, alongside the existing
fusion tests:

```bash
python -m pytest tests/unit/test_landsat_international.py \
  tests/unit/test_landsat_comparison.py tests/unit/test_pan_fusion.py -q
```

Public offline demonstration maps and source attributions are in the [tutorial guide](landsat-tutorials.md). Full research imagery, vectors and large/restricted overlay maps remain local and are not part of the proposed PR.
