"""
25 — 47 US States: Delineate-Anything v2 against the OpenET Field Boundaries (2021-2024)

A simple check across the conterminous US: Delineate-Anything v2 as
released, on NAIP and on Sentinel-2, in one 5 km square in each of the 47
states of the OpenET agricultural field boundaries, version 2.1.1 (Desert
Research Institute for the OpenET Consortium; every state of the
conterminous US except Alabama), evaluated against the field boundaries of
the square. One square per state shows how the released model does across
the country's farming landscapes; it is not an accuracy estimate for the
states.

Data. The OpenET field boundaries v2.1.1 are not public yet (they are to be
released with their data descriptor), so they are not in this repository and
the script reads a local copy: ``--openet-dir``, the ``OFB_DATA_DIR``
environment variable or ``OPENET_DIR`` (in the notebook) names the folder
that holds one zipped shapefile per state (``<ST>.zip``, EPSG:4326;
``<ST>_README.md`` describes the 71 attributes). ``SOURCECODE`` names the
agency delivery each outline comes from: in most states the USDA Farm
Service Agency's Common Land Units (CLU), elsewhere state and federal
inventories (the tables below give each square's mix). In New Mexico and
Nevada the script uses the DRI field boundaries instead of the delivery's
files: the DRI/NMOSE field boundaries (V18; Desert Research Institute and
New Mexico Office of the State Engineer) in
``examples/NM_Field_Boundaries.zip``, as in examples 12, 20 and 21, and the
DRI field boundaries (Nevada, 2026-08-20; Desert Research Institute) in
``examples/NV_Field_Boundaries.zip``, as in example 24. Neither file is in
this repository: without its zip the state is skipped with a message. Every
file is read in place with GDAL's ``/vsizip/``, and a run reads only the
square's bounding box; nothing is extracted or copied. Figures and
documentation may show the reference outlines of every state (the OpenET
outlines, CLU fields included, and the New Mexico and Nevada DRI
boundaries) and where the squares lie, with the attribution below.

Study squares, one per state, chosen by a fixed rule from the field
boundaries and the 2024 USDA Cropland Data Layer (CDL) alone, before any
model ran on them (``rule_square``):

    - candidates: the cells of the 5 km grid of each WGS 84 UTM zone that
      the state reaches into (corners at multiples of 5,000 m) whose centre
      lies in that zone and which lie entirely inside the state (U.S. Census
      Bureau 2023 cartographic boundary, 1:500,000, pinned by SHA-256, with
      vertices added about 100 m apart before it is projected; this
      keeps the squares off Alabama, which is not in the delivery, and off
      Canada and Mexico, whose fields are not mapped);
    - the square is the candidate with the most farmed fields: fields of
      the state's file (any source) of at least 2,500 m2 (in EPSG:6933, as
      agribound's area filter computes it) whose representative point (in
      the zone's coordinates) lies in the cell and on a pixel that CDL 2024
      maps as a crop, hay or pasture class (``FARMED_CDL``; Earth Engine's
      ``USDA/NASS/CDL/2024``, band ``cropland``, sampled on its own 10 m
      grid); ties go to the southernmost, then the westernmost cell centre.

Farmed CDL classes (``FARMED_CDL``; the 2024 legend in Earth Engine, which
matches NASS's list of CDL codes, checked on 2026-10-08): 1-61 (field
crops, vegetables, alfalfa (36) and other hay (37), clover/wildflowers
(58), sod/grass seed (59), switchgrass (60) and fallow/idle cropland (61)),
66-77 (tree crops, vines and Christmas trees), 204-254 (more crops and the
double crops) and 176 (Grass/Pasture); the ranges include some codes that
the legend does not use. NASS made 176 in January 2014, when it recoded the
CDLs of 1997-2013, by merging the classes pasture/grass (62), grassland
herbaceous (171) and pasture/hay (181), and it is one of the land-cover
classes (111-195) that CDL trains on the National Land Cover Database
rather than on FSA's field records, so it holds rangeland as well as
pasture: the rule counts it as farmed. Not farmed: 63-65 (forest,
shrubland, barren), 81-83 (clouds/no data, developed, water), 87 and 88
(wetlands, non-agricultural/undefined), 92 (aquaculture), 111 and 112 (open
water, perennial ice/snow), 121-124 (developed), 131 (barren), 141-143
(forest), 152 (shrubland), 190 and 195 (wetlands) and 0 (no class). Code 62
has not been used since the 2013 CDL.
The OpenET files' own crop type, ``CROP_2024``, is not used: it is the
majority class of the field's 30 m CDL pixels, with Other Hay/Non Alfalfa
(37) on most fields that CDL 2024 maps as Grass/Pasture (176), and in
California a crosswalk of the state's 2023 crop map. On a sample of 8,800
fields (200 of at least 2,500 m2 in each of the other 44 states), it equals
the CDL class at the representative point for 59 % of them, and the farmed
or not farmed call agrees for 93 %.

The rule favours squares of many small farmed fields, so the squares are not
typical of their states. The script runs on the squares recorded in
``STATES``; ``--select`` re-applies the rule and reports whether it gives
the recorded squares, for example with another release of the field
boundaries or of CDL. It reads every field of the states' files
(``SELECT_WORKERS`` states at a time) and the Census outlines, and it
samples CDL 2024 in Earth Engine at the fields of the cells it examines: the
cells in decreasing order of their number of fields, until no later cell
can hold more farmed fields than the best one so far.

The squares (``--select`` on 2026-10-09, run twice with the same result;
centres in degrees; County: Census 2023, + when the square reaches into
another county; NAIP flight: the dates of the NAIP images used, see Runs):

    ST  Zone  Corner (m)         Centre (lon, lat)  County            NAIP flight
    AR  15N   685000, 3825000    -90.9560, 34.5722  Phillips          2023-10-07
    AZ  11N   720000, 3605000   -114.6296, 32.5826  Yuma              2023-06-17
    CA  11N   275000, 4050000   -119.4872, 36.5921  Fresno+           2022-06-24
    CO  13N   540000, 4230000   -104.5143, 38.2395  Pueblo            2023-09-20
    CT  19N   250000, 4650000    -71.9877, 41.9856  Northeastern CT   2023-07-12, 08-31
    DE  18N   480000, 4260000    -75.2007, 38.5107  Sussex            2023-06-02
    FL  17N   385000, 3090000    -82.1437, 27.9528  Hillsborough      2023-01-06
    GA  16N   685000, 3770000    -84.9680, 34.0765  Polk+             2023-09-30, 10-07
    IA  15N   675000, 4675000    -90.8490, 42.2295  Jackson           2023-08-31
    ID  11N   680000, 4715000   -114.7758, 42.5883  Twin Falls        2023-07-01, 07-12
    IL  15N   730000, 4680000    -90.1812, 42.2601  Jo Daviess        2023-07-09, 07-10
    IN  16N   445000, 4220000    -87.5992, 38.1488  Vanderburgh+      2022-06-15
    KS  15N   275000, 4425000    -95.6053, 39.9684  Brown             2023-08-16
    KY  16N   660000, 4160000    -85.1593, 37.5952  Marion            2022-06-18
    LA  15N   700000, 3300000    -90.9041, 29.8364  Lafourche+        2023-05-01
    MA  18N   695000, 4690000    -72.6018, 42.3597  Hampshire         2023-08-23
    MD  18N   465000, 4250000    -75.3723, 38.4202  Wicomico          2023-06-02
    ME  19N   585000, 5190000    -67.8516, 46.8803  Aroostook         2023-11-06
    MI  16N   595000, 4770000    -85.8019, 43.0990  Ottawa+           2022-08-16
    MN  15N   620000, 4830000    -91.4814, 43.6354  Houston           2023-07-08
    MO  15N   445000, 4320000    -93.6067, 39.0500  Lafayette         2022-06-28
    MS  16N   345000, 3830000    -88.6635, 34.6230  Prentiss          2023-08-23
    MT  12N   375000, 5385000   -112.6625, 48.6286  Glacier           2023-07-22
    NC  18N   370000, 4030000    -76.4224, 36.4292  Pasquotank        2022-06-15, 06-18
    ND  13N   670000, 5115000   -102.7646, 46.1892  Adams             2023-07-06
    NE  14N   690000, 4580000    -96.6982, 41.3710  Saunders          2022-07-10
    NH  19N   305000, 4800000    -71.3752, 43.3507  Merrimack+        2023-08-23
    NJ  18N   505000, 4475000    -74.9116, 40.4481  Hunterdon         2023-08-19 to 09-01
    NM  13N   340000, 3845000   -106.7208, 34.7575  Valencia          2022-06-04
    NV  11N   340000, 4365000   -118.8303, 39.4427  Churchill         2022-07-01
    NY  18N   545000, 4575000    -74.4322, 41.3476  Orange            2022-07-04
    OH  17N   440000, 4465000    -81.6771, 40.3561  Coshocton+        2023-05-18, 07-22
    OK  14N   590000, 4020000    -97.9692, 36.3431  Garfield          2023-05-08
    OR  10N   550000, 5030000   -122.3287, 45.4441  Clackamas+        2022-07-19
    PA  18N   310000, 4420000    -77.1943, 39.9319  Adams             2022-04-24, 05-10
    RI  19N   285000, 4625000    -71.5566, 41.7709  Providence        2023-07-12, 08-22
    SC  17N   600000, 3750000    -79.8913, 33.9079  Florence+         2023-04-18
    SD  13N   625000, 4945000   -103.3915, 44.6695  Butte             2023-08-09, 08-15
    TN  16N   300000, 3960000    -89.1853, 35.7868  Crockett          2023-10-18
    TX  14N   615000, 2900000    -97.8236, 26.2373  Cameron           2022-06-05
    UT  12N   455000, 4375000   -111.4946, 39.5461  Sanpete           2021-11-07
    VA  18N   395000, 4050000    -76.1462, 36.6125  Chesapeake+       2023-10-09
    VT  18N   630000, 4870000    -73.3475, 43.9937  Addison           2023-06-20, 06-22
    WA  10N   705000, 5145000   -120.2982, 46.4491  Yakima            2023-07-13
    WI  15N   685000, 4850000    -90.6691, 43.8018  Monroe            2022-06-27 to 08-31
    WV  18N   255000, 4360000    -77.8153, 39.3779  Jefferson         2022-07-22
    WY  12N   645000, 4795000   -109.1810, 43.3159  Fremont           2022-07-10

Their reference fields (Fields: the fields of the state's file of at least
2,500 m2 by representative point; Farmed: those that the rule counts;
Dropped: the smaller fields left out; Median: the median size of the
reference fields in EPSG:6933; Sources: ``SOURCECODE``, share of the
fields; the commonest CDL 2024 classes at their representative points, with
the developed, forest and wetland classes merged, pasture: Grass/Pasture
(176), hay: Other Hay/Non Alfalfa (37), wheat/soy: the double crop):

    ST  Farmed  Fields  Dropped  Median  Sources (share)       Commonest CDL 2024 classes
    AR     232     269        3  4.9 ha  CLU 100 %             soybeans, cotton, wetlands
    AZ     515     832       10  1.2 ha  LCRVPD 95 %, CLU 5 %  shrubland, alfalfa, citrus
    CA     782     796       11  1.6 ha  CADWR_2023prov 100 %  peaches, grapes, citrus
    CO     420     505       10  1.3 ha  CODSS 100 %           pasture, alfalfa, corn
    CT     280     313        3  1.5 ha  CLU 100 %             pasture, corn, hay
    DE     462     541       10  2.0 ha  CLU 100 %             corn, soybeans, developed
    FL     169     231        1  2.7 ha  FLSAID 100 %          strawberries, pasture, developed
    GA     215     237        7  2.8 ha  CLU 100 %             pasture, corn, cotton
    IA     667     669       37  1.1 ha  CLU 100 %             corn, soybeans, pasture
    ID     595     631       13  1.8 ha  IDWRSP 92 %, CLU 8 %  pasture, corn, alfalfa
    IL     452     458       20  1.3 ha  CLU 100 %             corn, soybeans, pasture
    IN     323     353        5  4.1 ha  CLU 100 %             soybeans, corn, wheat/soy
    KS     338     344       13  1.5 ha  CLU 100 %             soybeans, corn, pasture
    KY     385     444       19  2.9 ha  CLU 100 %             pasture, soybeans, corn
    LA   1,071   1,132        8  1.2 ha  CLU 100 %             sugarcane, soybeans, fallow
    MA     427     492        9  1.3 ha  CLU 100 %             potatoes, corn, pasture
    MD     553     618        1  1.8 ha  CLU 100 %             corn, soybeans, developed
    ME     262     288        5  3.1 ha  CLU 100 %             potatoes, pasture, barley
    MI     289     341        4  2.7 ha  CLU 100 %             apples, corn, soybeans
    MN     507     507       13  1.1 ha  CLU 100 %             corn, soybeans, pasture
    MO     356     371       24  2.2 ha  CLU 100 %             soybeans, corn, pasture
    MS     197     205        2  3.3 ha  CLU 100 %             pasture, soybeans, forest
    MT     678     684        2  1.9 ha  CLU 100 %             pasture, winter wheat, spring wheat
    NC     713     829        4  1.7 ha  CLU 100 %             soybeans, corn, developed
    ND     435     440        4  2.6 ha  CLU 100 %             canola, spring wheat, corn
    NE     238     246        5  2.7 ha  CLU 80 %, NEDNR 20 %  soybeans, corn, pasture
    NH     174     206        1  1.9 ha  CLU 100 %             pasture, hay, developed
    NJ     319     340        1  2.8 ha  CLU 100 %             pasture, soybeans, corn
    NM     443     829      171  0.6 ha  V18                   pasture, developed, shrubland
    NV     185     238        1  4.5 ha  DRI 100 %             alfalfa, shrubland, wetlands
    NY     617     668        3  1.7 ha  CLU 100 %             onions, soybeans, pasture
    OH     497     513       10  0.9 ha  CLU 100 %             pasture, corn, soybeans
    OK     171     172        6  3.4 ha  CLU 100 %             winter wheat, pasture, wheat/soy
    OR     545     617        4  1.4 ha  DRI 100 %             pasture, developed, forest
    PA     334     377        7  1.9 ha  CLU 100 %             pasture, hay, soybeans
    RI      62      82        0  1.4 ha  CLU 100 %             pasture, sweet corn, developed
    SC     337     408       16  2.2 ha  CLU 100 %             corn, soybeans, cotton
    SD     282     285        3  3.9 ha  CLU 100 %             pasture, alfalfa, corn
    TN     444     515       18  1.9 ha  CLU 100 %             soybeans, pasture, cotton
    TX     278     329        0  3.6 ha  CLU 100 %             pasture, sorghum, developed
    UT     619     752       18  1.9 ha  UT_DWR 100 %          pasture, alfalfa, shrubland
    VA     780     936        3  1.5 ha  CLU 100 %             soybeans, corn, wheat/soy
    VT     512     589       24  2.1 ha  CLU 100 %             pasture, hay, corn
    WA     451     535       14  1.8 ha  WSDA 100 %            apples, grapes, cherries
    WI     852     870       17  0.9 ha  CLU 100 %             corn, pasture, alfalfa
    WV     218     249       21  4.4 ha  CLU 100 %             pasture, soybeans, corn
    WY      79     152        2  5.2 ha  WYSWP 100 %           shrubland, pasture, alfalfa

``MOD_DATE`` (the OpenET files' date of a field's last geometry change) is
one date per square, the state's delivery: 2020-08-17 (CO, IA, ID, KS, MO,
ND, OK, SD, TX, WY), 2020-08-31 (AZ, MT), 2021-04-16 (NE), 2021-04-21 (the
other 28 squares with an OpenET file: CLU, and FLSAID in Florida),
2023-12-15 (OR) and 2023-12-19 (UT, WA), apart from 10 fields dated
2024-01-01 (1 in CO, 7 in WI, 2 in WY); the California, New Mexico and
Nevada files have none.

The tie-break decides three squares: Kentucky's (385 farmed fields, as many
as the cell at 675000, 4165000 of the same zone), West Virginia's (218, as
at 250000, 4365000) and Wyoming's (79, as at 640000, 4800000). New
Jersey's, Texas's, Oklahoma's, Georgia's, Idaho's, Maine's and Minnesota's
squares win by 1, 2, 5, 6, 6, 6 and 6 farmed fields. Another release of the
field boundaries or of CDL can move these first.

Why farmed fields: with every field counted, the densest cell of some states
lies in a town, on rangeland or in a wetland. In New Mexico it is in
Albuquerque (Bernalillo and Sandoval counties: 1,015 fields, 250 of them
farmed, most of the others on developed land or shrubland), in Nevada in the
Carson Valley (Douglas County: 289 fields, 132 farmed, the others mostly
wetland and shrubland), in Florida in Miami-Dade (236, 93 farmed) and in
California in Riverside County (1,069, 680 farmed, 352 on shrubland).
Counting farmed fields instead changes the square of 17 states, these four
among them (New Mexico's lies 51 km south, in Valencia County, and Nevada's
in Churchill County, mostly alfalfa). Pasture (176) is still the commonest
class in 19 squares, and in Oregon's (Clackamas County) CDL maps 539 of the
617 fields as Grass/Pasture. In New Mexico the square also holds 171 V18
fields under 2,500 m2, which the evaluation drops. New Mexico's count, 443
farmed fields, is the one with PROJ's transformation grids
(``PROJ_NETWORK=ON``, which the 1.1 run set). Without them the NAD83 V18
fields are placed 0.15-0.16 m from where the grids put them, one
representative point falls on the Grass/Pasture side of a CDL 2024 pixel
edge instead of on developed land, and the count is 444 (the square is the
same).

Runs (each a separate agribound run with ``lulc_filter=False``):

    - Delineate-Anything v2 (``large_v2``), used as released, on NAIP (1 m;
      native 0.3 or 0.6 m) of the latest year whose 4-band images in Earth
      Engine's USDA/NAIP/DOQQ cover the square completely (``STATES``;
      queried on 2026-10-09: 2021 in Utah, 2022 in 16 states, 2023 in 30;
      NAIP is flown every two to three years per state). In Oregon and West
      Virginia that is 2022: Oregon's square has no later images, and the
      2023 images cover 48 % of West Virginia's. Utah's images (7 November
      2021) and Maine's (6 November 2023) were taken after the harvest;
    - the same model on Sentinel-2 (10 m): a median of 1 April to 31 October
      2024 (``S2_SEASON``; the Sentinel-2 cloud mask keeps snow, so a
      calendar-year median can hold winter snow in the northern states),
      with up to 1,000 detections per tile, as in examples 23 and 24 (at
      10 m one tile covers 2.56 km);
    - with ``--crop-sequence`` (opt-in), crop-history zones from the USDA
      Cropland Data Layer: ``source="cdl-sequence"``, ``year=2023``,
      ``crop_sequence_years=8`` (CDL 2016-2023, read at 30 m on the square's
      UTM grid; one Earth Engine request for the image list and one
      download) and ``engine="crop-sequence"`` with its defaults (the farmed
      classes of ``FARMED_CDL`` in more than half of the years, developed
      land (121-124) as a barrier, a per-year focal mode of radius 3 px,
      regions of equal sequence, regions under 1 ha merged into their
      largest neighbour; CPU only, no model). It is scored against the same
      reference by the same rules;
    - with ``--sam-auto`` (opt-in): the direct SAM engine
      (``engine="sam-auto"``, SAM 2.1 large automatic masks) on the same NAIP
      and Sentinel-2 composites, with the engine's own per-source defaults
      (chosen on two Nevada development squares; Sentinel-2's comes with a
      WARNING about its GPU cost and its result), scored by the same rules.
      Its rows go to ``summary_sam_auto.csv``, so ``summary.csv`` keeps one
      Delineate-Anything row per state and source. These squares, NAIP and
      Sentinel-2 composites were the engine's pre-registered evaluation
      (2026-10-13), whose primary results leave out California (the training
      data of Delineate-Anything v2 hold California's DWR labels). On NAIP
      the engine's F1 was 0.017 higher than Delineate-Anything's on the
      other 46 squares (95 % interval 0.004 to 0.030; both near F1 0.3). On
      Sentinel-2 it did not transfer: its F1 was 0.152 lower (95 % interval
      0.127 to 0.178 lower; lower on 45 of the 46 squares), and it took 17.0
      times Delineate-Anything's GPU time per km2 on two large New Mexico
      boxes.

The crop-sequence zones are zones of uniform 2016-2023 crop history, made
the way USDA NASS makes its Crop Sequence Boundaries; they are not field
outlines. Adjacent fields with the same crop sequence fall into one zone, and
fields narrower than about 60-90 m or smaller than 1 ha are absorbed or lost.
In the crop-sequence pilot (2026-10-09, research code, ten of these squares:
LA, AZ, VA, CA, MT, FL, KY, WY, WI and IA) they scored F1 0.006-0.084 against
these references, against 0.011-0.649 for Delineate-Anything on a Sentinel-2
2024 composite with agribound's defaults; many neighbouring CLU fields share
a crop sequence (0.18-0.67 of the adjacent pairs in the six CLU squares).
agribound's engine, run as this option runs it on the same ten squares
(2026-10-10), scored F1 0.004-0.078. With this option on all 47 squares
(agribound 1.1.0, 2026-10-11), the zones scored F1 from 0.004 (Montana) to
0.259 (Indiana), median 0.080, against a median of 0.314 for
Delineate-Anything on NAIP and 0.326 on Sentinel-2.

Evaluation (``agribound.evaluate``): one-to-one matching at IoU >= 0.5. The
reference is every field of the state's file (any source) whose
representative point lies inside the square, without those smaller than
2,500 m2 (the predictions' minimum area; the script prints how many it
drops), and the predictions are selected by the same rule (agribound's
default ``aoi_selection``). Read the scores with these limits:

    - the sources map different things. A CLU is a farm-programme unit (in
      FSA's definition, a contiguous piece of land with one owner, one
      producer and one land cover and management, bounded by permanent
      features such as fences, roads and waterways), so one CLU can hold
      several fields that the imagery separates, and the reverse; several
      sources map irrigated land only (in these squares IDWRSP, CODSS,
      WYSWP, NEDNR and FLSAID), and the Nevada DRI file maps irrigated
      fields and fields retired from irrigation (example 24), so fields that
      the model finds outside them count as false positives;
    - the outlines are each agency's delivery, not a map of the imagery
      year (the OpenET files' ``MOD_DATE``, documented as the date of a
      field's last geometry change, is printed per square; in these squares
      it is one date per state delivery, 2020-08-17 to 2023-12-19, apart
      from 10 fields dated 2024-01-01 in Colorado, Wisconsin and Wyoming),
      so fields that changed in between count as errors; NAIP (2021-2023) is
      also older than the Sentinel-2 composites (2024);
    - Delineate-Anything v2 was trained on FBIS-73M, whose labelled US
      training patches are the California DWR / Land IQ crop maps. In its
      public lists (revision 9e4cdbd; the US- and Canadian-prefixed patches,
      located by their own GeoTIFF headers on 2026-10-13), 128 labelled
      training patches intersect California's square, so the California
      scores are not a held-out test, and no patch intersects any of the
      other 46 squares. The nearest labelled training patch lies 15.1 km
      from Arizona's square, 102.0 km from Nevada's and at least 169.3 km
      from the others; unlabelled training patches lie 7.7 km from
      Arizona's square and 10.5 km from Wyoming's. Only the US- and
      Canadian-prefixed patches were located, and the public lists were
      published after training and changed on the day of the release, so
      they may not be exactly what the released weights saw;
    - in Montana and North Dakota most reference fields are CLU strips about
      50 m wide (86 % and 80 % of the fields). The 2005 NAIP shows crop and
      fallow strips of that width and orientation in both squares (in
      Montana's, the CLU lines lie a median 7-8 m from their edges in 2004
      and 2005, and 15-19 m in one block); by the 2023 NAIP both squares are
      farmed in blocks of many strips. On Montana's 2023 composite, 10 % of
      the 1,163 CLU lines that two fields share show contrast between the
      fields' pixels (5 % would by chance), and 3 of the 684 fields have a
      visible edge on every shared side (3 more share no side), so a
      delineation of exactly what the image shows would score at most about
      F1 0.01-0.08 against this reference (over seven contrast thresholds);
      North Dakota's square shows the same change (contrast across 62 % of
      its shared CLU lines in 2005, 17 % in 2023). The rule picked the most
      subdivided cell of Montana (86 % of its fields are strips, against 23 %
      of Montana's fields), so the Montana score says little about the
      state's farmland. These measurements of the reference and the imagery
      (no model; 2026-10-12) were made after the scores were known and
      changed nothing here; the documentation gallery shows the Montana
      square on the NAIP of 2005 and 2023.

Outputs (``outputs/multistate_openet/``): per state (``<st>/``) the square
(``study_area.geojson``), the reference fields it is evaluated against
(``reference.gpkg``, with each field's CDL 2024 class, ``cdl_2024``, for
maps such as the documentation gallery's), the two prediction layers (three
with ``--crop-sequence``: the zones, ``zones_cdl-sequence_2016-2023.gpkg``;
two more with ``--sam-auto``, ``fields_<source>_sam-auto_<year>.gpkg``),
their metrics and a map with the reference outlines; ``summary.csv`` with one
row per state and run, years and farmed fields (``n_reference_farmed``)
included (a run replaces the rows of the states and runs it scores and keeps
the others, so a run with ``--only`` keeps the other states, and a run
without ``--crop-sequence`` keeps the zones' rows), and with ``--sam-auto``
``summary_sam_auto.csv``, the same for the direct SAM runs. agribound reuses an
output whose provenance record matches the run's settings, so a second run
only re-evaluates; ``--overwrite`` recomputes.

Attribution: OpenET agricultural field boundaries, version 2.1.1, Desert
Research Institute for the OpenET Consortium (2026), with the field
boundaries of their source agencies (names and providers from the data
descriptor's source registry):

    - Common Land Unit (CLU), USDA Farm Service Agency: 35 squares, and
      part of Arizona's, Idaho's and Nebraska's;
    - Statewide Crop Mapping, California Department of Water Resources
      (``CADWR_2023prov``): California;
    - Agricultural Land Use, Washington State Department of Agriculture
      (``WSDA``): Washington;
    - Desert Research Institute field delineation, Desert Research
      Institute (``DRI``): Oregon;
    - Water Related Land Use, Utah Division of Water Resources
      (``UT_DWR``): Utah;
    - Eastern Snake River Plain irrigated lands, Idaho Department of Water
      Resources (``IDWRSP``): Idaho;
    - Colorado Decision Support System irrigated lands, Colorado Division
      of Water Resources (``CODSS``): Colorado;
    - Wyoming irrigated lands, Wyoming State Water Plan (``WYSWP``): Wyoming;
    - Nebraska certified irrigated acres, Nebraska Department of Natural
      Resources (``NEDNR``): Nebraska;
    - Florida Statewide Agricultural Irrigation Demand, Florida Department
      of Agriculture and Consumer Services (``FLSAID``): Florida;
    - Lower Colorado River Valley boundaries, U.S. Bureau of Reclamation /
      P. Donnelly (``LCRVPD``): Arizona;

and, in place of the delivery's New Mexico and Nevada files, the DRI/NMOSE
field boundaries (V18), Desert Research Institute and New Mexico Office of
the State Engineer, and the DRI field boundaries (Nevada, 2026-08-20),
Desert Research Institute.

``--only CA UT`` runs some states, ``--only west`` (or ``midwest``,
``south``, ``northeast``: the Census regions) a region. ``--crop-sequence``
adds the crop-history zones run to every state (one small Earth Engine
download and a few CPU seconds per state). ``--sam-auto`` adds the two
direct SAM runs to every state (measured in the engine's pre-registered
evaluation on these squares, Apple M2 Max with MPS: the pipeline's
delineate step, model loading included, took 3,965 s on NAIP and 321 s on
Sentinel-2 for the 47 states, about 84 s and 7 s per square; scoring adds
to that). ``--dry-run`` reads
the squares' fields and prints the squares, the reference counts and every
run's configuration, and runs and writes nothing (no Earth Engine).
``--select`` re-applies the square rule and runs nothing else (it samples
CDL 2024 in Earth Engine and downloads the Census outlines into
``outputs/multistate_openet/``). A run samples CDL 2024 at the reference
fields too, for the farmed count of each square.

Runtime: 64 minutes (3,815 s; the script printed 63.4 minutes) for the 94
runs in the development run of agribound 1.1 (2026-10-10, Apple M2 Max with
MPS; composites downloaded, the model weights already cached): about 1.8
minutes to read the references and sample CDL, 46 minutes for the 47 NAIP
runs (49 to 73 s each; 23 of the 46 minutes to download the composites of
about 5,000 x 5,000 px), 16 minutes for the 47 Sentinel-2 runs (13 to 45 s
each, mostly the download) and under a minute for the evaluations and maps.
``--crop-sequence`` adds a few seconds per state (1.1 to 3.8 s per zones
run): 4.5 minutes (272 s) with agribound 1.1.0 on 2026-10-11, in a run that
reused the 94 outputs and scored every run again. Measured on 2026-10-08 and
2026-10-09 on the same machine: ``--select`` 2.5 to 6 minutes (the 47 files,
4.7 GB of zips, and the CDL samples; the first read of a cloud-synced copy
downloads it), ``--dry-run`` about 2 minutes.

Prerequisites:
    pip install "agribound[gee,delineate-anything]"
    (with --crop-sequence also "agribound[crop-sequence]", with --sam-auto also
    "agribound[sam-auto]")
    agribound auth --project YOUR_GEE_PROJECT
    The OpenET field boundaries v2.1.1 (local copy, not public yet):
        --openet-dir DIR or OFB_DATA_DIR=DIR
    New Mexico and Nevada (local data, not in this repository; read in place):
        examples/NM_Field_Boundaries.zip and examples/NV_Field_Boundaries.zip
    Run from the repository root:
        python examples/25_multistate_openet_field_boundaries.py
"""

import argparse
import hashlib
import io
import json
import logging
import math
import os
import sys
import time
import urllib.request
import warnings
from pathlib import Path

import agribound
from agribound.provenance import to_jsonable

# Paths below are relative to the repository root. The notebook version of this
# script runs from examples/notebooks/, so it changes to the repository root first.
if Path.cwd().name == "notebooks" and Path.cwd().parent.name == "examples":
    os.chdir(Path.cwd().parents[1])

warnings.filterwarnings("ignore", category=FutureWarning, module=r"geedim\..*")
warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"geedim\..*")
# GDAL corrects the ring order of some OpenET polygons as it reads them, and says so (from
# the reading threads of --select as "Warning 1: ... invalid winding order" lines).
warnings.filterwarnings("ignore", message=".*invalid winding order.*", category=RuntimeWarning)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logging.getLogger("googleapiclient").setLevel(logging.CRITICAL)
logging.getLogger("httpx").setLevel(logging.WARNING)

# --- Configuration ---
DATA_ENV = "OFB_DATA_DIR"  # environment variable: the folder of the <ST>.zip files
#: The folder of the <ST>.zip files when neither --openet-dir nor $OFB_DATA_DIR gives it
#: (for example in the notebook); None: one of them must.
OPENET_DIR = None
DATA_LABEL = "OpenET agricultural field boundaries v2.1.1"
OPENET_COLUMNS = ["Unique_ID", "SOURCECODE", "MOD_DATE", "CROP_2024"]
#: New Mexico and Nevada: the DRI field boundaries instead of the delivery's files (local
#: data, not in this repository; read in place; a state whose zip is missing is skipped).
DRI_REFERENCES = {
    "NM": {
        "zip": "examples/NM_Field_Boundaries.zip",
        "member": "Master_NM_Field_Boundaries_V18.shp",
        "label": "DRI/NMOSE field boundaries (V18)",
        "code": "V18",  # the file has no SOURCECODE
        "columns": ["id", "CDL_Crop_T"],
    },
    "NV": {
        "zip": "examples/NV_Field_Boundaries.zip",
        "member": "nevada_agricultural_field_boundaries_20260820.gpkg",
        "label": "DRI field boundaries (Nevada, 2026-08-20)",
        "code": "DRI",
        "columns": ["NWI_ID", "SOURCECODE", "CROP_2024"],
    },
}
OUTPUT_DIR = Path("outputs/multistate_openet")
SIDE_M = 5000  # side of the squares, and spacing of the grid of candidate squares
MIN_AREA = 2500  # m^2 (EPSG:6933); the predictions' default minimum area, also for the reference
#: The 2024 USDA Cropland Data Layer in Earth Engine (band "cropland"; 10 m, EPSG:5070).
CDL_ASSET = "USDA/NASS/CDL/2024"
#: CDL classes that count as farmed: crop, hay and pasture classes (see the docstring).
FARMED_CDL = frozenset([*range(1, 62), *range(66, 78), *range(204, 255), 176])
CDL_BATCH = 5000  # points per Earth Engine request
IOU_THRESHOLD = 0.5
IMAGERY = {"naip": "NAIP", "sentinel2": "Sentinel-2"}  # the two runs of every state
#: The opt-in run (--crop-sequence): crop-history zones from CDL 2016-2023.
CROP_SEQUENCE_SOURCE = "cdl-sequence"
CROP_SEQUENCE_YEAR = 2023  # the last CDL year
CROP_SEQUENCE_YEARS = 8
#: Names of the runs in the tables.
RUN_LABELS = {**IMAGERY, CROP_SEQUENCE_SOURCE: "CDL zones"}
#: The opt-in runs (--sam-auto) of the direct SAM engine on the IMAGERY composites, with the
#: engine's own per-source defaults; their rows go to SAM_AUTO_SUMMARY, not to summary.csv.
SAM_AUTO = "sam-auto"
SAM_AUTO_SUMMARY = "summary_sam_auto.csv"
S2_YEAR = 2024
S2_SEASON = ("04-01", "10-31")  # Sentinel-2 composite window (month-day, inclusive)
S2_MAX_DETECTIONS = 1000  # per tile; at 10 m one 256 px tile covers 2.56 km
SELECT_WORKERS = 4  # --select reads this many states at a time
#: State outlines for the square rule: U.S. Census Bureau 2023 cartographic boundaries,
#: 1:500,000, pinned by SHA-256.
CENSUS_STATES_URL = "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_state_500k.zip"
CENSUS_STATES_SHA256 = "4a9b4f5cf993cd23738ac49b58fbb556f1f097fcf5e404a9dc10348dd41f7432"
#: The four Census regions (``--only`` takes their names as well as state codes).
REGIONS = {
    "west": ("AZ", "CA", "CO", "ID", "MT", "NM", "NV", "OR", "UT", "WA", "WY"),
    "midwest": ("IA", "IL", "IN", "KS", "MI", "MN", "MO", "ND", "NE", "OH", "SD", "WI"),
    "south": (
        "AR", "DE", "FL", "GA", "KY", "LA", "MD", "MS", "NC", "OK", "SC", "TN", "TX", "VA", "WV",
    ),
    "northeast": ("CT", "MA", "ME", "NH", "NJ", "NY", "PA", "RI", "VT"),
}  # fmt: skip
#: Each state's square, chosen by the rule (``--select``): (EPSG code of the UTM zone,
#: lower-left corner x, y in metres), and the latest year of the NAIP images in Earth
#: Engine that cover it completely.
STATES = {
    "AR": ((32615, 685000, 3825000), 2023),
    "AZ": ((32611, 720000, 3605000), 2023),
    "CA": ((32611, 275000, 4050000), 2022),
    "CO": ((32613, 540000, 4230000), 2023),
    "CT": ((32619, 250000, 4650000), 2023),
    "DE": ((32618, 480000, 4260000), 2023),
    "FL": ((32617, 385000, 3090000), 2023),
    "GA": ((32616, 685000, 3770000), 2023),
    "IA": ((32615, 675000, 4675000), 2023),
    "ID": ((32611, 680000, 4715000), 2023),
    "IL": ((32615, 730000, 4680000), 2023),
    "IN": ((32616, 445000, 4220000), 2022),
    "KS": ((32615, 275000, 4425000), 2023),
    "KY": ((32616, 660000, 4160000), 2022),
    "LA": ((32615, 700000, 3300000), 2023),
    "MA": ((32618, 695000, 4690000), 2023),
    "MD": ((32618, 465000, 4250000), 2023),
    "ME": ((32619, 585000, 5190000), 2023),
    "MI": ((32616, 595000, 4770000), 2022),
    "MN": ((32615, 620000, 4830000), 2023),
    "MO": ((32615, 445000, 4320000), 2022),
    "MS": ((32616, 345000, 3830000), 2023),
    "MT": ((32612, 375000, 5385000), 2023),
    "NC": ((32618, 370000, 4030000), 2022),
    "ND": ((32613, 670000, 5115000), 2023),
    "NE": ((32614, 690000, 4580000), 2022),
    "NH": ((32619, 305000, 4800000), 2023),
    "NJ": ((32618, 505000, 4475000), 2023),
    "NM": ((32613, 340000, 3845000), 2022),
    "NV": ((32611, 340000, 4365000), 2022),
    "NY": ((32618, 545000, 4575000), 2022),
    "OH": ((32617, 440000, 4465000), 2023),
    "OK": ((32614, 590000, 4020000), 2023),
    "OR": ((32610, 550000, 5030000), 2022),
    "PA": ((32618, 310000, 4420000), 2022),
    "RI": ((32619, 285000, 4625000), 2023),
    "SC": ((32617, 600000, 3750000), 2023),
    "SD": ((32613, 625000, 4945000), 2023),
    "TN": ((32616, 300000, 3960000), 2023),
    "TX": ((32614, 615000, 2900000), 2022),
    "UT": ((32612, 455000, 4375000), 2021),
    "VA": ((32618, 395000, 4050000), 2023),
    "VT": ((32618, 630000, 4870000), 2023),
    "WA": ((32610, 705000, 5145000), 2023),
    "WI": ((32615, 685000, 4850000), 2022),
    "WV": ((32618, 255000, 4360000), 2022),
    "WY": ((32612, 645000, 4795000), 2022),
}


def openet_dir(args):
    """The folder of the state zips: ``--openet-dir``, else $OFB_DATA_DIR, else OPENET_DIR."""
    folder = args.openet_dir or os.environ.get(DATA_ENV) or OPENET_DIR
    if not folder:
        raise SystemExit(
            f"The {DATA_LABEL} are not public yet; this example reads a local copy. Pass "
            f"--openet-dir DIR or set {DATA_ENV}=DIR, where DIR holds one <ST>.zip per state."
        )
    folder = Path(folder).expanduser()
    if not folder.is_dir():
        raise SystemExit(f"Folder of the {DATA_LABEL} not found: {folder}")
    return folder


def reference_path(folder, state):
    """GDAL path of a state's field boundaries inside their zip (read in place)."""
    if state in DRI_REFERENCES:
        spec = DRI_REFERENCES[state]
        return f"/vsizip/{spec['zip']}/{spec['member']}"
    archive = Path(folder) / f"{state}.zip"
    if not archive.is_file():
        raise SystemExit(f"{archive} not found: the {DATA_LABEL} need one <ST>.zip per state.")
    return f"/vsizip/{archive}/{state}.shp"


def reference_label(state):
    """Name of a state's field boundaries."""
    return DRI_REFERENCES[state]["label"] if state in DRI_REFERENCES else DATA_LABEL


def region_of(state):
    """The Census region of *state*."""
    return next(name for name, states in REGIONS.items() if state in states)


def cell(x, y):
    """The SIDE_M square with lower-left corner (*x*, *y*) as a shapely box."""
    from shapely.geometry import box

    return box(x, y, x + SIDE_M, y + SIDE_M)


def reproject(area, epsg, crs):
    """*area* (a polygon in *epsg*) in *crs*, with its edges densified every 100 m."""
    import geopandas as gpd

    return gpd.GeoSeries([area.segmentize(100)], crs=epsg).to_crs(crs).iloc[0]


def rep_point_inside(gdf, area):
    """Rows whose representative point lies inside *area* (same CRS)."""
    return gdf[gdf.representative_point().within(area).to_numpy()].copy()


def census_outlines(save=True):
    """The Census state outlines (EPSG:4269) by state code, checked by SHA-256.

    Downloaded once into OUTPUT_DIR; with *save* False a missing copy is only read into
    memory (``--dry-run`` writes nothing).
    """
    import geopandas as gpd

    path = OUTPUT_DIR / Path(CENSUS_STATES_URL).name
    if path.exists():
        data = path.read_bytes()
    else:
        with urllib.request.urlopen(CENSUS_STATES_URL) as response:
            data = response.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != CENSUS_STATES_SHA256:
        raise SystemExit(f"{CENSUS_STATES_URL}: SHA-256 {digest}, expected {CENSUS_STATES_SHA256}")
    if save and not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return gpd.read_file(io.BytesIO(data)).set_index("STUSPS")


def cdl_classes(points):
    """The CDL 2024 class under each point of *points* (a GeoSeries), 0 where CDL has none.

    Samples Earth Engine's CDL_ASSET on its own grid (10 m, EPSG:5070), CDL_BATCH points
    per request; the classes come back in the order of the points.
    """
    import ee
    import numpy as np

    if not len(points):
        return np.zeros(0, dtype=int)
    lonlat = points.to_crs(4326).get_coordinates().to_numpy()
    image = ee.Image(CDL_ASSET).select("cropland").unmask(0)
    classes = []
    for start in range(0, len(lonlat), CDL_BATCH):
        batch = ee.List(
            [[float(lon), float(lat)] for lon, lat in lonlat[start : start + CDL_BATCH]]
        )
        sites = ee.FeatureCollection(batch.map(lambda xy: ee.Feature(ee.Geometry.Point(xy))))
        sampled = image.reduceRegions(sites, ee.Reducer.first(), crs=image.projection())
        classes += sampled.aggregate_array("first").getInfo()
    if len(classes) != len(lonlat):
        raise RuntimeError(f"{CDL_ASSET}: {len(classes)} classes for {len(lonlat)} points")
    return np.asarray(classes, dtype=int)


def farmed(classes):
    """True where a CDL class is in FARMED_CDL."""
    import numpy as np

    return np.isin(classes, sorted(FARMED_CDL))


def rule_square(path, outline):
    """The square the rule picks in a state, its farmed fields and its fields.

    Returns ``((epsg, x, y), n_farmed, n_fields)`` (the EPSG code of the UTM zone and the
    lower-left corner), or ``(None, 0, 0)`` when no cell qualifies. *path* is the state's
    field boundaries (:func:`reference_path`) and *outline* its polygon from
    :func:`census_outlines`. Reads every field of the file and samples CDL 2024 (Earth
    Engine) at the representative points of the fields of the cells it examines: the
    cells in decreasing order of their number of fields, until no later cell can hold
    more farmed fields than the best one so far (see the docstring).
    """
    import geopandas as gpd
    import numpy as np
    import pyogrio
    import shapely

    from agribound.postprocess import filter_polygons

    fields = filter_polygons(pyogrio.read_dataframe(path, columns=[]), min_area_m2=MIN_AREA)
    lon_min, _, lon_max, _ = outline.bounds
    candidates = []  # (-fields, centre latitude, centre longitude, epsg, x, y)
    points = {}  # per zone: the representative points and the corners of their cells
    for zone in range(int((lon_min + 180) // 6) + 1, int((lon_max + 180) // 6) + 2):
        epsg = 32600 + zone
        zone_points = fields.geometry.to_crs(epsg).representative_point()
        zone_points = zone_points[~zone_points.is_empty & zone_points.notna()]
        xy = np.floor(zone_points.get_coordinates().to_numpy() / SIDE_M).astype(np.int64)
        points[epsg] = (zone_points, xy * SIDE_M)
        corners, counts = np.unique(xy * SIDE_M, axis=0, return_counts=True)
        centres = gpd.GeoSeries(gpd.points_from_xy(*(corners + SIDE_M / 2).T), crs=epsg)
        centres = centres.to_crs(4326)
        in_zone = (np.floor((centres.x.to_numpy() + 180) / 6) + 1) == zone
        for (x, y), n, lon, lat in zip(
            corners[in_zone], counts[in_zone], centres.x[in_zone], centres.y[in_zone], strict=True
        ):
            candidates.append((-int(n), lat, lon, epsg, int(x), int(y)))

    inside = {}  # the state's outline in each zone's coordinates
    best = None  # (-farmed fields, centre latitude, centre longitude, epsg, x, y, fields)
    for neg_n, lat, lon, epsg, x, y in sorted(candidates):
        if best is not None and -neg_n < -best[0]:
            break  # this cell and all later ones have fewer fields than the best has farmed
        if epsg not in inside:
            # Densified first (vertices at most 0.001 degrees, about 100 m, apart): a long
            # straight border, such as the 49th parallel, is not straight in UTM.
            densified = outline.segmentize(0.001)
            inside[epsg] = gpd.GeoSeries([densified], crs=4269).to_crs(epsg).iloc[0]
            shapely.prepare(inside[epsg])
        if not inside[epsg].contains(cell(x, y)):
            continue
        zone_points, corners = points[epsg]
        in_cell = zone_points[(corners[:, 0] == x) & (corners[:, 1] == y)]
        n_farmed = int(farmed(cdl_classes(in_cell)).sum())
        key = (-n_farmed, lat, lon, epsg, x, y, -neg_n)  # ties: south, then west
        if best is None or key < best:
            best = key
    if best is None:
        return None, 0, 0
    return best[3:6], -best[0], best[6]


def select_squares(folder, states):
    """Re-apply the square rule; print each state's square and whether it is the recorded one."""
    from concurrent.futures import ThreadPoolExecutor

    import geopandas as gpd

    outlines = census_outlines()
    start = time.time()
    # The area filter logs one line per state, unlabelled when states are read together.
    logging.getLogger("agribound.postprocess.filter").setLevel(logging.WARNING)
    print(
        f"\nSquare rule ({SIDE_M / 1000:g} km cells, farmed fields (CDL 2024) of at least "
        f"{MIN_AREA:,} m2; {len(states)} states, {SELECT_WORKERS} at a time):"
    )

    def apply_rule(state):
        return rule_square(reference_path(folder, state), outlines.loc[state, "geometry"])

    with ThreadPoolExecutor(SELECT_WORKERS) as pool:
        for state, (found, n_farmed, n) in zip(states, pool.map(apply_rule, states), strict=True):
            if found is None:
                print(f"  {state}: no cell qualifies")
                continue
            epsg, x, y = found
            centre = gpd.GeoSeries([cell(x, y).centroid], crs=epsg).to_crs(4326).iloc[0]
            recorded = STATES[state][0]
            status = "the recorded square" if recorded == found else f"recorded: {recorded}"
            print(
                f"  {state}: EPSG:{epsg} corner ({x}, {y}), centre {centre.x:.4f}, "
                f"{centre.y:.4f}; {n_farmed:,} farmed of {n:,} fields; {status}"
            )
    print(f"  ({(time.time() - start) / 60:.1f} minutes)")


def load_square(folder, state, square):
    """A state's square and its reference fields.

    Returns ``(area, reference, n_small)``: the square in its UTM zone, the reference
    fields in the same CRS and the number of fields under MIN_AREA dropped. Reads only
    the fields that reach into the square's bounding box.
    """
    import pyogrio

    from agribound.postprocess import filter_polygons

    epsg, x, y = square
    area = cell(x, y)
    path = reference_path(folder, state)
    bbox = reproject(area, epsg, pyogrio.read_info(path)["crs"]).bounds
    columns = DRI_REFERENCES[state]["columns"] if state in DRI_REFERENCES else OPENET_COLUMNS
    fields = pyogrio.read_dataframe(path, bbox=bbox, columns=columns)
    inside = rep_point_inside(fields.to_crs(epsg), area)
    reference = filter_polygons(inside, min_area_m2=MIN_AREA).reset_index(drop=True)
    return area, reference, len(inside) - len(reference)


def source_mix(state, reference):
    """The sources of the reference fields and their shares, e.g. ``IDWRSP 79 %, CLU 21 %``."""
    if not len(reference):
        return "-"
    if "SOURCECODE" not in reference:  # V18: one source
        return f"{DRI_REFERENCES[state]['code']} 100 %"
    shares = reference["SOURCECODE"].fillna("(none)").value_counts(normalize=True)
    percent = {code: "<1" if s < 0.005 else round(100 * s) for code, s in shares.items()}
    return ", ".join(f"{code} {value} %" for code, value in percent.items())


def describe(state, square, area, reference, n_small):
    """Print a state's reference, square and reference fields."""
    import geopandas as gpd
    import pandas as pd

    epsg, x, y = square
    print(f"\n{state} ({region_of(state)}): {reference_label(state)}")
    centre = gpd.GeoSeries([area.centroid], crs=epsg).to_crs(4326).iloc[0]
    print(
        f"  square: EPSG:{epsg} corner ({x}, {y}), centre {centre.x:.4f}, {centre.y:.4f}; "
        f"NAIP {STATES[state][1]}, Sentinel-2 {S2_YEAR}"
    )
    size_ha = reference.to_crs(6933).area / 1e4
    if "cdl_2024" in reference:  # sampled in a run (Earth Engine), not in a dry run
        n_farmed = f"{int(farmed(reference['cdl_2024']).sum()):,} farmed (CDL 2024)"
    else:
        n_farmed = "farmed fields not counted in a dry run"
    print(
        f"  reference: {len(reference):,} fields ({n_farmed}), median {size_ha.median():.1f} ha; "
        f"{n_small:,} smaller than {MIN_AREA:,} m2 dropped"
    )
    line = f"  sources (share of fields): {source_mix(state, reference)}"
    if "MOD_DATE" in reference:  # the OpenET files: date of the last geometry change
        dates = pd.to_datetime(reference["MOD_DATE"].dropna())
        if dates.empty:
            line += "; MOD_DATE not recorded"
        elif dates.min() == dates.max():
            line += f"; MOD_DATE {dates.min():%Y-%m-%d}"
        else:
            line += (
                f"; MOD_DATE median {dates.median():%Y-%m-%d}, "
                f"{dates.min():%Y-%m-%d} to {dates.max():%Y-%m-%d}"
            )
    print(line)


def study_area(epsg, area, state):
    """A GeoJSON FeatureCollection of a state's square in EPSG:4326."""
    from shapely.geometry import mapping

    feature = {
        "type": "Feature",
        "geometry": mapping(reproject(area, epsg, 4326)),
        "properties": {"name": f"{state} square ({reference_label(state)})"},
    }
    return {"type": "FeatureCollection", "features": [feature]}


def run_config(state, source, args, engine="delineate-anything"):
    """AgriboundConfig of one run: Delineate-Anything v2 as released on *source*, or (for
    CROP_SEQUENCE_SOURCE) the crop-history zones of CDL 2016-2023; with ``engine=SAM_AUTO``
    the direct SAM engine on the same composite, with no engine_params."""
    from agribound.config import AgriboundConfig

    out_dir = OUTPUT_DIR / state.lower()
    if source == CROP_SEQUENCE_SOURCE:
        first = CROP_SEQUENCE_YEAR - CROP_SEQUENCE_YEARS + 1
        return AgriboundConfig(
            study_area=str(out_dir / "study_area.geojson"),
            source=source,
            year=CROP_SEQUENCE_YEAR,
            crop_sequence_years=CROP_SEQUENCE_YEARS,
            engine="crop-sequence",
            output_path=str(out_dir / f"zones_{source}_{first}-{CROP_SEQUENCE_YEAR}.gpkg"),
            gee_project=args.gee_project,
            lulc_filter=False,
            overwrite=args.overwrite,
        )
    if source == "naip":
        year, settings = STATES[state][1], {}
    else:
        year = S2_YEAR
        settings = {
            "date_range": (f"{year}-{S2_SEASON[0]}", f"{year}-{S2_SEASON[1]}"),
            "engine_params": {"max_detections": S2_MAX_DETECTIONS},
        }
    label = "da"
    if engine == SAM_AUTO:  # its own per-source defaults: no engine_params
        settings.pop("engine_params", None)
        label = SAM_AUTO
    return AgriboundConfig(
        study_area=str(out_dir / "study_area.geojson"),
        source=source,
        year=year,
        engine=engine,
        output_path=str(out_dir / f"fields_{source}_{label}_{year}.gpkg"),
        gee_project=args.gee_project,
        lulc_filter=False,
        overwrite=args.overwrite,
        **settings,
    )


def runs(args):
    """The sources of a state's runs: IMAGERY, and CROP_SEQUENCE_SOURCE with --crop-sequence."""
    return list(IMAGERY) + ([CROP_SEQUENCE_SOURCE] if args.crop_sequence else [])


def state_runs(args):
    """(source, engine) of a state's runs: those of runs(), then with --sam-auto the direct
    SAM runs on the IMAGERY composites."""
    engines = {CROP_SEQUENCE_SOURCE: "crop-sequence"}
    default = [(source, engines.get(source, "delineate-anything")) for source in runs(args)]
    return default + ([(source, SAM_AUTO) for source in IMAGERY] if args.sam_auto else [])


def run_year(state, source):
    """The year of a run: NAIP's latest complete year, S2_YEAR or the last CDL year."""
    if source == "naip":
        return STATES[state][1]
    return CROP_SEQUENCE_YEAR if source == CROP_SEQUENCE_SOURCE else S2_YEAR


def run_dates(source, year):
    """The dates a run reads (``date_range`` in the summary; empty for the NAIP mosaic)."""
    if source == "sentinel2":
        return f"{year}-{S2_SEASON[0]} to {year}-{S2_SEASON[1]}"
    if source == CROP_SEQUENCE_SOURCE:
        return f"{year - CROP_SEQUENCE_YEARS + 1}-{year}"
    return ""


def dry_run(states, args):
    """Print every run's configuration (validated; nothing is run or written)."""
    print(f"\n{'=' * 70}\nDry run: configurations (nothing is run or written)\n{'=' * 70}")
    for state in states:
        for source, engine in state_runs(args):
            config = run_config(state, source, args, engine)
            if config.date_range or config.source == CROP_SEQUENCE_SOURCE:
                window = run_dates(config.source, config.year)
            else:  # NAIP: one mosaic, the year's images on top of the neighbouring years'
                window = f"mosaic, {config.year} on top"
            print(
                f"  {state} {config.source:<9} {config.engine} {config.year} {window:<24} "
                f"{config.engine_params or ''} -> {config.output_path}"
            )


def fmt(value, digits=2):
    """Format a metric (NaN-safe)."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "-"
    return "-" if math.isnan(value) else f"{value:.{digits}f}"


def print_summary(rows):
    """The per-state table: reference and predicted fields, P, R, F1 and mean IoU."""
    print(
        f"\n{'State':<5} {'Main source':<20} {'Imagery':<10} {'Year':>4} {'Ref':>5} "
        f"{'Pred':>5} {'TP':>5} {'P':>5} {'R':>5} {'F1':>5} {'IoU':>5}"
    )
    for row in rows:
        start = (
            f"{row['state']:<5} {row['sources'].split(', ')[0]:<20} "
            f"{RUN_LABELS[row['imagery']]:<10} "
            f"{row['year']:>4} {row['n_reference']:>5}"
        )
        if row["status"] != "ok":
            print(f"{start} {row['status']}")
            continue
        print(
            f"{start} {row['n_predicted']:>5} {row['tp']:>5} {fmt(row['precision']):>5} "
            f"{fmt(row['recall']):>5} {fmt(row['f1']):>5} {fmt(row['iou_mean']):>5}"
        )
    print(
        f"\nP, R, F1: one-to-one matching at IoU >= {IOU_THRESHOLD:g}; IoU: mean IoU of the "
        f"matched pairs. Reference: fields of at least {MIN_AREA:,} m2 by representative point. "
        "Main source: the commonest SOURCECODE of the reference fields and its share."
    )


def merged_summary(path, rows):
    """The summary table for *path*: this run's *rows*, and the rows of the earlier file at
    *path* (if any) of every state and run that this run did not score (other states with
    ``--only``, and the zones' rows of a run without ``--crop-sequence``), in the order of
    STATES."""
    import pandas as pd

    summary = pd.DataFrame(rows)
    if Path(path).exists():
        kept = pd.read_csv(path)
        scored = set(zip(summary["state"], summary["imagery"], strict=True))
        keep = [pair not in scored for pair in zip(kept["state"], kept["imagery"], strict=True)]
        summary = pd.concat([kept[keep], summary], ignore_index=True)
    order = {state: i for i, state in enumerate(STATES)}
    return summary.sort_values("state", key=lambda col: col.map(order), kind="stable")


def show_in_notebook(web_map):
    """Display *web_map* inline when this file runs as a Jupyter notebook."""
    if "ipykernel" in sys.modules:
        from IPython.display import display

        display(web_map)


def parse_args(argv=None):
    """Parse command-line arguments (none are read inside Jupyter)."""
    parser = argparse.ArgumentParser(
        description=f"47 US states: Delineate-Anything v2 against the {DATA_LABEL}."
    )
    parser.add_argument(
        "--openet-dir",
        default=None,
        help=f"Folder of the {DATA_LABEL}, one <ST>.zip per state (default: ${DATA_ENV}, "
        "then OPENET_DIR in the script).",
    )
    parser.add_argument(
        "--gee-project",
        default=None,
        help=(
            "Earth Engine project ID (default: $GEE_PROJECT, then the gcloud project, then "
            "the project_id of the $AGRIBOUND_GEE_SERVICE_ACCOUNT_KEY or "
            "$GOOGLE_APPLICATION_CREDENTIALS file)."
        ),
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        metavar="ST_OR_REGION",
        help=f"Run only these states (codes such as CA) or Census regions ({', '.join(REGIONS)}).",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Recompute outputs that already exist."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the squares, the reference counts and every run's configuration; run "
        "and write nothing (no Earth Engine).",
    )
    parser.add_argument(
        "--crop-sequence",
        action="store_true",
        help="Also run the crop-history zones of CDL 2016-2023 (source cdl-sequence, engine "
        "crop-sequence) in every state, scored like the other runs (needs scikit-image: "
        'pip install "agribound[crop-sequence]").',
    )
    parser.add_argument(
        "--sam-auto",
        action="store_true",
        help="Also run the direct SAM engine (engine sam-auto, SAM 2.1 automatic masks) with "
        "its per-source defaults on the NAIP and Sentinel-2 composites of every state, scored "
        f"like the other runs; its rows go to {SAM_AUTO_SUMMARY} (needs the sam-auto extra). "
        "In its pre-registered evaluation on these squares it did not transfer to Sentinel-2 "
        "(F1 0.15 below Delineate-Anything's; see the docstring).",
    )
    parser.add_argument(
        "--select",
        action="store_true",
        help="Re-apply the square rule over each whole state and report whether it gives "
        "the recorded squares; run nothing else (downloads the Census state outlines and "
        "samples CDL 2024 in Earth Engine).",
    )
    if argv is None and "ipykernel" in sys.modules:
        argv = []  # Jupyter passes its own kernel arguments in sys.argv
    args = parser.parse_args(argv)
    if args.only:  # state codes and region names -> states, in the order of STATES
        chosen = set()
        for name in args.only:
            if name.lower() in REGIONS:
                chosen.update(REGIONS[name.lower()])
            elif name.upper() in STATES:
                chosen.add(name.upper())
            else:
                parser.error(
                    f"--only: {name!r} is neither a state ({', '.join(STATES)}) nor a "
                    f"region ({', '.join(REGIONS)})"
                )
        args.only = [state for state in STATES if state in chosen]
    return args


def main():
    import geopandas as gpd

    from agribound.evaluate import evaluate
    from agribound.visualize import show_comparison

    args = parse_args()
    start_time = time.time()
    states = []
    for state in args.only or list(STATES):  # New Mexico and Nevada need the DRI files
        if state in DRI_REFERENCES and not Path(DRI_REFERENCES[state]["zip"]).is_file():
            spec = DRI_REFERENCES[state]
            print(
                f"{state} skipped: {spec['zip']} not found. It holds the {spec['label']}: "
                "local data, not in this repository (contact the author for access)."
            )
        else:
            states.append(state)
    if not states:
        raise SystemExit("No state to run.")
    folder = openet_dir(args) if any(s not in DRI_REFERENCES for s in states) else None
    for state in states:  # fail early on a missing file
        reference_path(folder, state)
    if not args.dry_run:  # CDL 2024 for the farmed fields (the runs use Earth Engine too)
        from agribound.auth import setup_gee

        setup_gee(project=args.gee_project)
    if args.select:
        select_squares(folder, states)
        return
    print(f"Reading the field boundaries of {', '.join(states)} ({folder or 'DRI files'})")
    squares = {}
    for state in states:
        square = STATES[state][0]
        area, reference, n_small = load_square(folder, state, square)
        if not args.dry_run:  # the CDL class at each representative point, as the rule reads it
            reference["cdl_2024"] = cdl_classes(reference.representative_point())
        squares[state] = (square, area, reference, n_small)
        describe(state, *squares[state])
    if args.dry_run:
        dry_run(states, args)
        return

    rows, sam_rows = [], []  # summary.csv; SAM_AUTO_SUMMARY (--sam-auto)
    for state in states:
        square, area, reference, n_small = squares[state]
        epsg = square[0]
        out_dir = OUTPUT_DIR / state.lower()
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "study_area.geojson").write_text(json.dumps(study_area(epsg, area, state)))
        if len(reference):  # the evaluated reference, for maps such as the gallery's
            reference.to_file(out_dir / "reference.gpkg", layer="reference")
        centre = gpd.GeoSeries([area.centroid], crs=epsg).to_crs(4326).iloc[0]
        layers = []
        for source, engine in state_runs(args):
            year = run_year(state, source)
            table = sam_rows if engine == SAM_AUTO else rows
            row = {
                "state": state,
                "region": region_of(state),
                "reference": reference_label(state),
                "sources": source_mix(state, reference),
                "imagery": source,
                "engine": engine,
                "year": year,
                "date_range": run_dates(source, year),
                "utm_epsg": epsg,
                "square_x": square[1],
                "square_y": square[2],
                "centre_lon": round(centre.x, 5),
                "centre_lat": round(centre.y, 5),
                "n_reference": len(reference),
                "n_reference_farmed": int(farmed(reference["cdl_2024"]).sum()),
                "n_reference_small_dropped": n_small,
            }
            sam = f" {SAM_AUTO}" if engine == SAM_AUTO else ""
            print(f"\n{'=' * 70}\n{state}: {RUN_LABELS[source]}{sam} {year}\n{'=' * 70}")
            try:
                config = run_config(state, source, args, engine)
                pred = agribound.delineate(config=config).to_crs(epsg)
            except Exception as exc:  # e.g. no imagery, a package missing
                print(f"  {state} {source}{sam} {year} failed: {type(exc).__name__}: {exc}")
                table.append({**row, "status": f"failed: {type(exc).__name__}"})
                continue
            metrics = evaluate(pred, reference, iou_threshold=IOU_THRESHOLD)
            table.append(
                {
                    **row,
                    "status": "ok",
                    "n_predicted": len(pred),
                    "tp": metrics["count_tp"],
                    "fp": metrics["count_fp"],
                    "fn": metrics["count_fn"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "iou_mean": metrics["iou_mean"],
                    "output": config.output_path,
                }
            )
            stem = Path(config.output_path).stem
            (out_dir / f"metrics_{stem}.json").write_text(
                json.dumps(
                    to_jsonable({"row": table[-1], "metrics": metrics}), indent=2, allow_nan=False
                )
            )
            kind = "zones" if source == CROP_SEQUENCE_SOURCE else "fields"
            print(f"  {state} {source}{sam} {year}: {len(pred)} {kind}, F1 {fmt(metrics['f1'])}")
            if source == CROP_SEQUENCE_SOURCE:
                label = f"Crop-history zones, CDL {run_dates(source, year)} (not field outlines)"
            elif engine == SAM_AUTO:
                label = f"SAM 2.1 automatic masks (sam-auto), {IMAGERY[source]} {year}"
            else:
                label = f"Delineate-Anything, {IMAGERY[source]} {year}"
            layers.append((pred, label, year))

        # The state's map: the prediction layers and the reference.
        if layers:
            years = sorted({year for _, _, year in layers})
            span = f"{years[0]}-{years[-1]}" if len(years) > 1 else str(years[0])
            map_path = out_dir / f"comparison_{span}.html"
            web_map = show_comparison(
                [frame for frame, _, _ in layers] + [reference],
                labels=[label for _, label, _ in layers] + [f"{state}: {reference_label(state)}"],
                output_html=str(map_path),
            )
            show_in_notebook(web_map)
            print(f"Map: {map_path}")

    # A run replaces the rows of the states and runs it scores and keeps the others.
    path = OUTPUT_DIR / "summary.csv"
    # convert_dtypes keeps the counts integers when a failed run leaves them empty
    merged_summary(path, rows).convert_dtypes().to_csv(path, index=False)
    print_summary(rows)
    print(f"Summary: {OUTPUT_DIR / 'summary.csv'}")
    if sam_rows:  # --sam-auto: the direct SAM runs, in their own table
        sam_path = OUTPUT_DIR / SAM_AUTO_SUMMARY
        merged_summary(sam_path, sam_rows).convert_dtypes().to_csv(sam_path, index=False)
        print("\nThe direct SAM engine (sam-auto) on the same composites:")
        print_summary(sam_rows)
        print(f"Summary of the direct SAM runs: {sam_path}")
    print(f"\nTotal runtime: {(time.time() - start_time) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
