# Finding and preparing stronger North American field references

The [six-site comparison](landsat-north-america-comparison.md) is complete,
but available crop/irrigation/declaration polygons are not independent
physical-field truth. This inventory distinguishes measured sites from
research leads and planned annotation. The goal is a small, dated, licensed
reference that labels **all visible cultivation units within known coverage**.

## Sources investigated beyond the selected sites

| Source | What was inspected | Decision / remaining requirement |
| --- | --- | --- |
| [USDA LTAR GIS 2020](https://doi.org/10.15482/USDA.ADC/1521161), [data access](https://ltar.ars.usda.gov/data/data-access/) | Downloaded public CC0 experiment-boundary GeoJSON and inspected attributes/geometries, including Upper Mississippi and research-plot layers. | Many units are experiments, farm/watershed footprints or generalized areas rather than individual cultivation fields. No score against these was substituted for a Corn Belt benchmark. Ask managers for dated whole-field layers and crop records. |
| [Oregon DRI/OWRD mapping](https://www.dri.edu/project/owrd-et/), [2025 technical report](https://www.oregon.gov/owrd/WRDReports/Huntington_et_al_2025_DRI_Report_41306.pdf) | Inspected provider release/report and linked geodatabase; historical maximum agricultural extent spans 1985–2022, with subdivisions used to reconcile annual irrigation history. | The [older Dryad record](https://datadryad.org/dataset/doi:10.5061/dryad.2v6wwpzvw) is explicitly superseded. New geodatabase not downloaded/inferred here. Qualify contemporary field geometry/year and redistribution terms before adopting it. |
| [USDA Crop Sequence Boundaries](https://www.nass.usda.gov/Research_and_Science/Crop-Sequence-Boundaries/index.php), [2024 metadata](https://www.nass.usda.gov/Research_and_Science/Crop-Sequence-Boundaries/metadata_Crop-Sequence-Boundaries-2024.htm) | Reviewed algorithm and source stack: synthetic polygons based on multiyear CDL. | Useful optional external product, not independent boundary truth; no CSB delineation/evaluation was run. Resampling historical 30 m CDL to a finer grid does not create measured detail. |
| [USDA CDL](https://data.nass.usda.gov/Research_and_Science/Cropland/sarsfaqs2.php), [AAFC Annual Crop Inventory](https://agriculture.canada.ca/atlas/apps/aef/cropinventory/index_en.html?AGRIAPP=24) | Crop-context products and class uncertainty. | Raster classifications, not physical-field polygons. Do not convert connected crop pixels into gold-standard boundaries or infer field-level species from imagery appearance. |
| [Mexico SIAP agricultural frontier](https://www.gob.mx/agricultura/dgsiap/documentos/frontera-agricola-de-mexico), [provider definition](https://www.gob.mx/agricultura/es/articulos/sabes-que-es-la-frontera-agricola?idiom=es) | Reviewed agricultural extent including recent cultivation/fallow potential land. | Does not establish contemporaneous individual fields. No suitable dated/licensed field reference accepted in this search; this is a documented gap, not a claim that none exists in Mexico. |
| [FSA geospatial customer services](https://www.fpacbc.usda.gov/geospatial-services/customer-services), [producer field exports](https://www.fsa.usda.gov/news-events/news/06-10-2024/usda-reminds-producers-file-crop-acreage-reports) | Current CLU access restrictions and producer access route. | Current nationwide CLU is not unrestricted public boundary truth. Seek a voluntary producer export with explicit permission; do not silently use an old release as current geometry. |

Dryland cereal systems and independent Corn Belt fields remain unmeasured.
Washington contains cereal groups but does not verify purely dryland fields.
No difficult comparison was replaced to improve scores. The reviewed LTAR
polygons did not meet the whole-field definition; the broad agricultural
extent in Mexico was not substituted. These gaps do not block the six
completed public-reference comparisons.

## Where to gather the missing information

Use public research/extension contacts to request small whole-field subsets,
without assuming that a layer is already open or that the organization will
share producer data. These pages identify appropriate organizations and
document relevant agricultural research:

| Priority gap | Organization / public contact route | Specific request |
| --- | --- | --- |
| Iowa/Minnesota Corn Belt maize–soybean rotations | [UMRB research locations](https://ltar.ars.usda.gov/sites/umrb/); [NLAE public contact](https://www.ars.usda.gov/midwest-area/ames/nlae/people/nlae-contact/) | Whole physical fields near an instrumented farm for a particular 2018–2023 season, separate from treatment plots and watershed boundaries; annual crop records and dated aerial/ground verification. |
| Wisconsin mixed annual/forage fields | [UW–Platteville Pioneer Farm and contact](https://www.uwplatt.edu/department/pioneer-farm) | Dated cultivation-field geometry for corn/oats/alfalfa systems, excluding experiment subdivisions; license and coverage declaration. |
| Arkansas rice / Mississippi soybean-cotton systems | [Lower Mississippi LTAR locations and leaders](https://ltar.ars.usda.gov/sites/lmrb/) | Individual on-farm cultivation units and crop/irrigation records, with landowner consent. The provider documents Jonesboro rice research and Stoneville soybean/cotton research. |
| Southeastern field systems | [Tifton Gulf Atlantic Coastal Plain LTAR](https://ltar.ars.usda.gov/sites/gacp/) | Whole-field geometry and independently verified crop/year records. Do not assume all GIS experiment/plot layers represent fields. |
| Washington survey vintage/license clarification | [WSDA Agricultural Land Use](https://agr.wa.gov/departments/land-and-water/natural-resources/agricultural-land-use) | Actual geometry-update date, local completeness, documented accuracy, and reuse/redistribution permission for the selected 2023 observations in the 2025 snapshot. |
| Montana physical edges / licensing | [DNRC service metadata](https://gis.dnrc.mt.gov/arcgis/rest/services/WRD/MontanaStatewideIrrigation/FeatureServer/0) | Whether pivot corners represent separate cultivation units, original mapping dates/accuracy, coverage omissions and redistribution terms. |
| Producer-managed fields elsewhere | Grower partners or local extension agents, using [farmers.gov acreage-report exports](https://www.fsa.usda.gov/news-events/news/06-10-2024/usda-reminds-producers-file-crop-acreage-reports) | Voluntary, deidentified geometry with written permission and crop/boundary dates. Keep private farm identifiers out of public provenance. |

No messages have been sent. Suggested request text:

> We are comparing agricultural field delineation from matched Landsat 8/9
> PAN, surface reflectance and image fusion in Agribound. Could you share a
> small, dated set of whole cultivation-field boundaries and crop records,
> or direct us to a public release, with permission for the intended
> analysis and redistribution?

Request CRS, acquisition/digitization/update dates, crop-label year,
mapping method, field versus treatment/declaration definitions, minimum
mapped size, omitted/inactive classes, positional accuracy, coverage area,
source imagery, satellite/model contributions, training overlap if known,
provider attribution, license version, and redistribution restrictions.
Ask for reference year matching available dated aerials and L8/9 imagery;
do not assume a publication year is the geometry year.

## Reproducible, prediction-blind manual reference workflow

This workflow and annotation kit are **prepared, not annotated or
independently reviewed datasets**. Independent reviewers are still required.
For US areas use [USGS NAIP](https://www.usgs.gov/centers/eros/science/usgs-eros-archive-aerial-photography-national-agriculture-imagery-program-naip)
through its dated archive/catalog or exact Earth Engine image IDs. For
Montérégie use [GeoMont spring 2023 orthophotos](https://www.donneesquebec.ca/recherche/dataset/geomont-orthophotographies-2023-region-de-la-monteregie),
20 cm imagery under CC BY 4.0. Record the actual tile dates, license and
product-specific positional accuracy. An undated web basemap is insufficient.

1. Choose 1–5 km² from reference availability, crop records and dated aerial
   coverage before running models. Freeze a WGS84 box, acquisition window,
   imagery IDs/checksums and selection rationale. For the missing Corn Belt
   and dryland sites, first confirm crop/year with the data partner.
2. Obtain a georeferenced dated aerial GeoTIFF. Record provider/item URL,
   actual acquisition date, CRS, pixel size, accuracy or `unknown`, license
   and SHA-256. The suite's overview JPEGs are qualitative inspection views,
   not substitutes for a georeferenced orthophoto used to digitize boundaries.
3. Create a **blank** GeoPackage in a local UTM CRS. For example, the
   already frozen Colusa area can be independently redigitized with this
   known July 2022 source:

   ```bash
   python examples/prepare_manual_field_reference.py --output outputs/manual_reference/colusa.gpkg --bbox -122.04 39.08 -121.99 39.12 --reference-year 2022 --imagery-source USDA/NAIP/DOQQ/m_3912157_nw_10_060_20220709 --imagery-date 2022-07-09 --imagery-license "USDA public domain" --positional-accuracy "Unknown for selected tiles"
   ```

   Use the other exact tile ID from the suite configuration for complete
   coverage and list both in metadata. Optional `--imagery-path PATH`
   records the downloaded GeoTIFF checksum. The helper refuses to overwrite
   an existing kit. It does not download, label, or certify reference data.
4. Open only the dated aerial, `evaluation_area`, and blank annotation
   layers in QGIS. Hide Agribound outputs, PAN/SR scores and CSB/model-derived
   boundaries. Digitize all visible cultivation units in the area, retaining
   whole geometry for fields whose representative point lies in the box.
   Follow a visible cultivation perimeter or a persistent shared physical
   edge. Do not split identical continuous cropping solely by ownership,
   declared crop code or experimental treatment. Do not equate every rice
   internal bund with a separate field: document the agreed field-unit rule
   before drawing. Avoid output-size filtering that would remove small
   reference fields and artificially improve evaluation.
5. Populate `field_id`, `edge_basis`, `confidence`, `annotator`,
   `imagery_date`, `crop_label`, `crop_evidence`, and `notes`. Crop labels
   are `unknown` unless independently documented. Record uncertain wetland,
   tree-shadow, fallow and continuous-crop interfaces in `ambiguous_edges`;
   exclude explicitly unresolved coverage rather than forcing a boundary.
6. Have a second person review all boundaries for a small benchmark,
   blind to model outputs. Record `reviewer`, `review_status` and disagreements.
   Check slivers, overlaps, duplicate IDs, missing eligible fields, CRS and
   positional offsets. Resolve disagreements using documented evidence;
   retain uncertainty instead of treating an arbitrary decision as exact.
7. Only after review, populate `reviewed_coverage` for areas with complete
   labeling, excluding unresolved areas. The template deliberately leaves
   this layer empty. Record review dates, eligibility/omission rules and
   license for the derived annotations. Mark metadata `reviewed` only after
   this work actually occurs; retain provider errors and boundary uncertainty.
8. Freeze the reference GeoPackage and metadata checksums. Add a new site
   with the `local` reference adapter and explicit license/vintage, using
   a new frozen suite/output directory. Extend the runner's reference
   preparation to read the independently reviewed coverage layer instead
   of the current union-of-known-footprints protocol. Keep the three matched
   experiments and tolerances unchanged. Annotation plans are not measured
   results and are not eligible for the completed-site tables.

The existing runs can support reference discovery; their predictions must
remain hidden from the annotators. An independently reviewed manual subset
would answer the physical-boundary question more directly than scoring
against modeled crop maps or declaration splits.
