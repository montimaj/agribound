# Citation and References

## Citing agribound

If you use agribound in your research, please cite the software (the DOI
below is the Zenodo concept DOI, which resolves to the latest version):

> Majumdar, S., Rapp, J., Huntington, J. L., ReVelle, P., Nozari, S., Smith, R. G., Hasan, M. F., Bromley, M., Atkin, J., Jensen, E. R., Ketchum, D., Abramowitz, J. C., & Roy, S. (2026). *Agribound: Unified agricultural field boundary delineation from satellite imagery using geospatial foundation models, pre-trained segmentation, and embeddings* (Version 1.0.1) [Software]. Zenodo. <https://doi.org/10.5281/zenodo.19229665>

The accompanying manuscript is in preparation:

> Majumdar, S., Rapp, J., Huntington, J. L., ReVelle, P., Nozari, S., Smith, R. G., Hasan, M. F., Bromley, M., Atkin, J., Jensen, E. R., Ketchum, D., & Roy, S. (2026). *Measuring what geospatial AI delivers for policy-grade agricultural field boundaries*. In prep. for *Remote Sensing of Environment*.

Machine-readable metadata: [`CITATION.cff`](https://github.com/montimaj/agribound/blob/main/CITATION.cff).

## References

Please also cite the models, datasets and tools you use. The entries below
were checked against Crossref, DataCite, the arXiv API or the publishers'
pages in September 2026; preprints without a peer-reviewed version are cited
as preprints.

### Delineate-Anything

Lavreniuk, M., Kussul, N., Shelestov, A., Yailymov, B., Salii, Y., Kuzin, V., & Szantoi, Z. (2025). Delineate Anything: Resolution-agnostic field boundary delineation on satellite imagery. European Conference on Artificial Intelligence (ECAI 2025). *arXiv:2504.02534*. <https://doi.org/10.48550/arXiv.2504.02534> (models `large`, `small`, dataset FBIS-22M)

Lavreniuk, M., Kussul, N., Shelestov, A., Salii, Y., Kuzin, V., Wang, C. J. L.-X., & Szantoi, Z. (2026). Delineate Anything v2: A global foundation model for field delineation. European Conference on Computer Vision Workshops (ECCVW 2026, GAIA workshop). *arXiv:2607.19069*. <https://doi.org/10.48550/arXiv.2607.19069> (model `large_v2`, dataset FBIS-73M)

Lavreniuk, M., Kussul, N., Shelestov, A., Salii, Y., Kuzin, V., Skakun, S., & Szantoi, Z. (2025). Delineate Anything Flow: Fast, country-level field boundary detection from any source. *arXiv:2511.13417*. <https://doi.org/10.48550/arXiv.2511.13417> (the reference pipeline behind `backend="reference"`)

### Fields of The World (FTW)

Kerner, H., Chaudhari, S., Ghosh, A., Robinson, C., Ahmad, A., Choi, E., Jacobs, N., Holmes, C., Mohr, M., Dodhia, R., Lavista Ferres, J. M., & Marcus, J. (2025). Fields of The World: A machine learning benchmark dataset for global agricultural field boundary segmentation. *Proceedings of the AAAI Conference on Artificial Intelligence*, 39(27), 28151-28159. <https://doi.org/10.1609/aaai.v39i27.35034>

Muhawenayo, G., Robinson, C., Khanal, S., Fang, Z., Corley, I., Wollam, A., Gao, T., Strnad, L., Avery, R., Estes, L., Tárano, A. M., Jacobs, N., & Kerner, H. (2026). PRUE: A practical recipe for field boundary segmentation at scale. *arXiv:2603.27101*. <https://doi.org/10.48550/arXiv.2603.27101> (the `FTW_PRUE_*` models)

Robinson, C., Muhawenayo, G., Khanal, S., Fang, Z., Corley, I., Tárano, A. M., Estes, L., Marcus, J., Jacobs, N., Kerner, H., Becker-Reshef, I., & Lavista Ferres, J. M. (2026). The first global agricultural field boundary map at 10m resolution. *arXiv:2605.11055* (preprint). <https://doi.org/10.48550/arXiv.2605.11055> (the published FTW polygons read by `query_ftw`; dataset CC-BY-4.0)

### GeoAI and Mask R-CNN

Wu, Q. (2026). GeoAI: A Python package for integrating artificial intelligence with geospatial data analysis and visualization. *Journal of Open Source Software*, 11(118), 9605. <https://doi.org/10.21105/joss.09605>

He, K., Gkioxari, G., Dollár, P., & Girshick, R. (2017). Mask R-CNN. *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, 2980-2988. <https://doi.org/10.1109/ICCV.2017.322>

### DINOv3

Siméoni, O., Vo, H. V., Seitzer, M., Baldassarre, F., Oquab, M., Jose, C., Khalidov, V., Szafraniec, M., Yi, S., Ramamonjisoa, M., Massa, F., Haziza, D., Wehrstedt, L., Wang, J., Darcet, T., Moutakanni, T., Sentana, L., Roberts, C., Vedaldi, A., Tolan, J., Brandt, J., Couprie, C., Mairal, J., Jégou, H., Labatut, P., & Bojanowski, P. (2025). DINOv3. *arXiv:2508.10104*. <https://doi.org/10.48550/arXiv.2508.10104> (The DINOv3 weights are under the [DINOv3 License](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md), whose clause 1.b.ii requires publications to acknowledge the use of DINO Materials.)

### Prithvi-EO-2.0 and TerraTorch

Szwarcman, D., Roy, S., Fraccaro, P., Gíslason, Þ. E., Blumenstiel, B., Ghosal, R., de Oliveira, P. H., de Sousa Almeida, J. L., Sedona, R., Kang, Y., Chakraborty, S., Wang, S., Gomes, C., Kumar, A., Gaur, V., Truong, M., Godwin, D., Khallaghi, S., Lee, H., Hsu, C.-Y., Akbari Asanjan, A., Mujeci, B., Shidham, D., Balogun, R. O., Kolluru, V., Keenan, T., Arevalo, P., Li, W., Alemohammad, H., Olofsson, P., Mayer, T., Hain, C., Kennedy, R., Zadrozny, B., Bell, D., Cavallaro, G., Watson, C., Maskey, M., Ramachandran, R., & Bernabe Moreno, J. (2026). Prithvi-EO-2.0: A versatile multitemporal foundation model for Earth observation applications. *IEEE Transactions on Geoscience and Remote Sensing*, 64, 1-20. <https://doi.org/10.1109/TGRS.2025.3642610>

Gomes, C., Blumenstiel, B., de Sousa Almeida, J. L., de Oliveira, P. H., Fraccaro, P., Marti Escofet, F., Szwarcman, D., Simumba, N., Kienzler, R., & Zadrozny, B. (2025). TerraTorch: The geospatial foundation models toolkit. *IGARSS 2025 - 2025 IEEE International Geoscience and Remote Sensing Symposium*, 6364-6368. <https://doi.org/10.1109/IGARSS55030.2025.11243570>

### Embeddings

Feng, Z., Atzberger, C., Jaffer, S., Knezevic, J., Sormunen, S., Young, R., Lisaius, M. C., Immitzer, M., Jackson, T., Ball, J., Coomes, D. A., Madhavapeddy, A., Blake, A., & Keshav, S. (2026). TESSERA: Temporal embeddings of surface spectra for Earth representation and analysis. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 34818-34831. arXiv:2506.20380.

Brown, C. F., Kazmierski, M. R., Pasquarella, V. J., Rucklidge, W. J., Samsikova, M., Zhang, C., Shelhamer, E., Lahera, E., Wiles, O., Ilyushchenko, S., Gorelick, N., Zhang, L. L., Alj, S., Schechter, E., Askay, S., Guinan, O., Moore, R., Boukouvalas, A., & Kohli, P. (2025). AlphaEarth Foundations: An embedding field model for accurate and efficient global mapping from sparse label data. *arXiv:2507.22291*. <https://doi.org/10.48550/arXiv.2507.22291>. The AlphaEarth Foundations Satellite Embedding dataset is produced by Google and Google DeepMind (CC-BY 4.0).

### Segment Anything and samgeo

Ravi, N., Gabeur, V., Hu, Y.-T., Hu, R., Ryali, C., Ma, T., Khedr, H., Rädle, R., Rolland, C., Gustafson, L., Mintun, E., Pan, J., Alwala, K. V., Carion, N., Wu, C.-Y., Girshick, R., Dollár, P., & Feichtenhofer, C. (2025). SAM 2: Segment anything in images and videos. *International Conference on Learning Representations (ICLR 2025)*. arXiv:2408.00714.

Carion, N., Gustafson, L., Hu, Y.-T., Debnath, S., Hu, R., Suris, D., Ryali, C., Alwala, K. V., Khedr, H., Huang, A., Lei, J., Ma, T., Guo, B., Kalla, A., Marks, M., Greer, J., Wang, M., Sun, P., Rädle, R., Afouras, T., Mavroudi, E., Xu, K., Wu, T.-H., Zhou, Y., Momeni, L., Hazra, R., Ding, S., Vaze, S., Porcher, F., Li, F., Li, S., Kamath, A., Cheng, H. K., Dollár, P., Ravi, N., Saenko, K., Zhang, P., & Feichtenhofer, C. (2026). SAM 3: Segment anything with concepts. *International Conference on Learning Representations (ICLR 2026)*. arXiv:2511.16719. (The SAM 3 licence asks publications to acknowledge the use of SAM materials.)

Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A. C., Lo, W.-Y., Dollár, P., & Girshick, R. (2023). Segment anything. *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 3992-4003. <https://doi.org/10.1109/ICCV51070.2023.00371>

Wu, Q., & Osco, L. P. (2023). samgeo: A Python package for segmenting geospatial data with the Segment Anything Model (SAM). *Journal of Open Source Software*, 8(89), 5663. <https://doi.org/10.21105/joss.05663>

Osco, L. P., Wu, Q., de Lemos, E. L., Gonçalves, W. N., Ramos, A. P. M., Li, J., & Marcato Junior, J. (2023). The Segment Anything Model (SAM) for remote sensing applications: From zero to one shot. *International Journal of Applied Earth Observation and Geoinformation*, 124, 103540. <https://doi.org/10.1016/j.jag.2023.103540>

### Data and platforms

Gorelick, N., Hancher, M., Dixon, M., Ilyushchenko, S., Thau, D., & Moore, R. (2017). Google Earth Engine: Planetary-scale geospatial analysis for everyone. *Remote Sensing of Environment*, 202, 18-27. <https://doi.org/10.1016/j.rse.2017.06.031>

Claverie, M., Ju, J., Masek, J. G., Dungan, J. L., Vermote, E. F., Roger, J.-C., Skakun, S. V., & Justice, C. (2018). The Harmonized Landsat and Sentinel-2 surface reflectance data set. *Remote Sensing of Environment*, 219, 145-161. <https://doi.org/10.1016/j.rse.2018.09.002>

Brown, C. F., Brumby, S. P., Guzder-Williams, B., et al. (2022). Dynamic World, near real-time global 10 m land use land cover mapping. *Scientific Data*, 9, 251. <https://doi.org/10.1038/s41597-022-01307-4>

Pasquarella, V. J., Brown, C. F., Czerwinski, W., & Rucklidge, W. J. (2023). Comprehensive quality assessment of optical satellite imagery using weakly supervised video learning. *Proceedings of the IEEE/CVF CVPR Workshops*, 2125-2135. <https://doi.org/10.1109/CVPRW59228.2023.00206> (Cloud Score+)

U.S. Geological Survey (2024). Annual NLCD Collection 1 Science Products (ver. 1.2, June 2026). U.S. Geological Survey data release. <https://doi.org/10.5066/P94UXNTS>

Copernicus Climate Change Service (2019). Land cover classification gridded maps from 1992 to present derived from satellite observations. ECMWF Climate Data Store. <https://doi.org/10.24381/cds.006f2c9a>

Roy, S., Majumdar, S., & Swetnam, T. (2025). samapriya/awesome-gee-community-datasets: Community Catalog (3.9.0). Zenodo. <https://doi.org/10.5281/zenodo.17641528> (Annual NLCD and C3S assets used by the LULC filter)

Wu, Q. (2020). geemap: A Python package for interactive mapping with Google Earth Engine. *Journal of Open Source Software*, 5(51), 2305. <https://doi.org/10.21105/joss.02305>

### Evaluation

Clinton, N., Holt, A., Scarborough, J., Yan, L., & Gong, P. (2010). Accuracy assessment measures for object-based image segmentation goodness. *Photogrammetric Engineering & Remote Sensing*, 76(3), 289-299. <https://doi.org/10.14358/PERS.76.3.289>

Persello, C., & Bruzzone, L. (2010). A novel protocol for accuracy assessment in classification of very high resolution images. *IEEE Transactions on Geoscience and Remote Sensing*, 48(3), 1232-1244. <https://doi.org/10.1109/TGRS.2009.2029570>

Stehman, S. V., & Foody, G. M. (2019). Key issues in rigorous accuracy assessment of land cover products. *Remote Sensing of Environment*, 231, 111199. <https://doi.org/10.1016/j.rse.2019.05.018>

### HPC

Boerner, T. J., Deems, S., Furlani, T. R., Knuth, S. L., & Towns, J. (2023). ACCESS: Advancing innovation: NSF's Advanced Cyberinfrastructure Coordination Ecosystem: Services & Support. *Practice and Experience in Advanced Research Computing (PEARC '23)*, 173-176. <https://doi.org/10.1145/3569951.3597559>

Work that uses NSF ACCESS resources must include the acknowledgement at
<https://access-ci.org/about/acknowledging-access/>; system papers are listed
in `examples/hpc/README.md`.

---

## Funding

This work was supported by multiple funding sources. The **New Mexico Office of the State Engineer (NMOSE)** provided reference field boundary data and supported the development of agricultural water use mapping in New Mexico. The **Google Satellite Embeddings Dataset Small Grants Program** enabled the integration of pre-computed satellite embeddings for unsupervised field boundary delineation. Access to the **SPOT 6 and 7 archive on Google Earth Engine** was provided through the Google Trusted Tester opportunity. Additional support was provided by the **U.S. Army Corps of Engineers** and **The U.S. Department of Treasury/State of Nevada**. This work was also supported by the **NASA Water Resources Applications Program**, the **United States Geological Survey (USGS)** and **NASA Landsat Science Team**, the **USGS Water Resources Research Institute**, the **Desert Research Institute Maki Endowment**, and the **Windward Fund**.

---

## Acknowledgments

Agribound builds on the work of many open-source projects and research teams:

- The **Ultralytics** team for the YOLO ecosystem
- **Meta AI Research** for the Segment Anything models (SAM, SAM 2, SAM 3) and DINOv3
- The **Fields of The World** consortium and Hannah Kerner's group at Arizona State University
- **Mykola Lavreniuk** and co-authors for Delineate-Anything
- **Qiusheng Wu** for the GeoAI and samgeo Python packages
- **NASA** and **IBM Research** for the Prithvi geospatial foundation model and TerraTorch
- **Google DeepMind** for AlphaEarth satellite embeddings
- **Feng et al.** for the TESSERA foundation model embeddings
- The **Google Earth Engine** team for planetary-scale geospatial computing
- The **fiboa** community for the field boundary schema standard
- The **TorchGeo** team for geospatial deep learning data loaders and utilities
- The **Desert Research Institute (DRI)** for supporting this research

---

## Disclaimer

This software is preliminary or provisional and is subject to revision. No warranty, expressed or implied, is made by DRI, USGS, the U.S. Government, or any contributing organization as to the functionality of the software. Any use of trade, firm, or product names is for descriptive purposes only and does not imply endorsement by the U.S. Government. See [DISCLAIMER.md](https://github.com/montimaj/agribound/blob/main/DISCLAIMER.md) for full details.

## AI Usage Disclosure

Portions of this software were developed with the assistance of AI coding tools, including Anthropic's Claude. AI was used to accelerate code scaffolding, documentation drafting, and test generation. All AI-generated code was reviewed, tested, and validated by the human authors. The scientific methodology, architectural decisions, algorithm selection, and domain-specific implementations reflect the expertise and judgment of the authors.
