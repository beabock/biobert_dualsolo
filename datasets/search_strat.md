# Search Strategy Documentation

**Date:** September 11, 2025  
**Database:** Web of Science Core Collection  
**Purpose:** Curate training dataset for BioBERT trophic mode classification

---

## Search Queries

### Solo Trophic Mode (single lifestyle)
```
("obligate mycorrhizal" OR "strictly endophytic" OR "exclusive saprotroph") AND fungus
```
**Results:** 119 articles

### Dual Trophic Mode (multiple lifestyles)
```
("dual lifestyle" OR "facultative lifestyle" OR "dual trophic mode" OR "lifestyle switching" OR "endophyte-saprotroph" OR "plant-associated saprotroph") AND fungi
```
**Results:** 70 articles

**Total candidate articles:** 189

---

## Selection Criteria

From the 189 candidate articles, abstracts were manually reviewed and 56 were selected based on:

1. **Unambiguous trophic mode description** — Abstract explicitly states trophic mode classification
2. **English language** — Only English-language abstracts included
3. **No duplicates** — Manually verified no overlap between solo and dual searches

Abstracts without explicit trophic mode statements were excluded.

---

## Curation Process

1. Selected articles organized in **Zotero** by class (solo folder, dual folder)
2. Exported as BibTeX files (`solo.bib`, `dual.bib`)
3. Processed with `parse_bib.py` to extract abstracts and create `abstracts.csv`
4. Split into training (60%) and test (40%) sets with stratified sampling

---

## Final Dataset

| Class | Count | Description |
|-------|-------|-------------|
| Solo (0) | 28 | Single trophic mode only |
| Dual (1) | 28 | Multiple trophic modes |
| **Total** | **56** | |

---

## Additional Resources Consulted

- **FunGuild** (funguild.org) — Trophic mode definitions (saprotroph, symbiotroph, pathotroph)

---

## Raw Notes


Whole search history from Septmeber 2025 (When I put together the dataset). It's possible that some of these are relevant to the search methods.

https://arpha.pensoft.net/preview.php?document_id=29454
https://riojournal.com/view_document.php?id=176590&view_role=11
https://preprints.arphahub.com/article/176591
https://preprints.arphahub.com/article/176591/
https://github.com/beabock/biobert_dualsolo
https://preprints.arphahub.com/article/176591/list/18/
https://riojournal.com/view_document.php?id=176590&view_role=11&section=2
https://riojournal.com/view_version.php?round_user_id=661404&version_id=939288&id=176590&view_role=11
https://riojournal.com/view_document.php?id=176590&view_role=11&section=1
https://riojournal.com/view_document.php?id=176590&view_role=11&event_id=0
https://arpha.pensoft.eu/preview.php?document_id=29454
https://github.com/beabock/Sap_Sym/releases/tag/v1.0.0
https://arpha.pensoft.net/preview.php?document_id=29423
https://mail.google.com/mail/u/0/#inbox/FMfcgzQcqHNBfXntrLvBsJdBQLwXCgSJ
https://mail.google.com/mail/u/0/#inbox/FMfcgzQcqHNBfXntrMFPHBSpPjSSfvKg
https://mail.google.com/mail/u/0/#inbox/FMfcgzQcqHNBfXpDpQNnxWwnhTsxvkdJ
https://mail.google.com/mail/u/0/#inbox/FMfcgzQcqHNBfXntrLvBsJdBQLwXCgSJ?compose=CllgCJftvBdPBJVgMFRcCzhGbRzWMmQWZXcpQgGDgdDLDdlzHJfBPJPdNtHBcCHcwJSxbKRSgNB
https://mail.google.com/mail/u/0/#inbox/FMfcgzQcqHNBfXntrLvBsJdBQLwXCgSJ?compose=lLtBPchKhnxzRkjLtvJkmhdMdDldTJXHGWqxlZJWLpCBjtjRQpGKTBBLJJQTvcXLCvjhFmxM
https://mail.google.com/mail/u/0/#inbox/FMfcgzQcqHNBfXntrLvBsJdBQLwXCgSJ?compose=GTvVlcRzDDKjkVjBMbtZBnPgZxChRsRCsxLcmVclFDtLqDlpZkVPPJCtbddGqWjsTMDZGfCwHkBSB
file:///C:/Users/beabo/OneDrive%20-%20Northern%20Arizona%20University/NAU/biobert_dualsolo/mycologia_brief_report.html
https://elicit.com/notebook/e7cc2a46-9ef4-4ede-bcbb-21e2528d76bd#186b5e0db835a7d3b54dc3e88c2a99cb
https://elicit.com/notebook/e7cc2a46-9ef4-4ede-bcbb-21e2528d76bd
https://www.catastrophicreations.com/blogs/articles/years-best-cat-toys-reviewed-by-experts?srsltid=AfmBOoo6EgqTVAf-TAnRlWeggJzeDtLZ2uZ1pI8IId_MQG2BVnY1ivQf
https://www.funguild.org/query.php?qText=Saprotroph&qDB=funguild_db&qField=trophicMode
https://elicit.com/notebook/0fe98d8a-9046-4c88-a273-b199e0c4cd2d
https://www.funguild.org/query.php?qText=symbiotroph&qDB=funguild_db&qField=trophicMode
https://www.funguild.org/query.php?qText=pathotroph&qDB=funguild_db&qField=trophicMode
https://www.funguild.org/query.php?qText=saprotroph&qDB=funguild_db&qField=trophicMode
https://www.funguild.org/query.php?qText=sapro&qDB=funguild_db&qField=trophicMode
https://elicit.com/notebook/cb672564-1264-453b-99e6-d6ab63d4d6f8
https://mail.google.com/mail/u/0/?tab=wm#inbox/QgrcJHsbdwdpbMqZmMRjlqRfxfNhcLVLBGq
https://www.sciencedirect.com/science/article/pii/S1754504822000022?via=ihub
https://www.sciencedirect.com/science/article/abs/pii/S1754504822000022?via%3Dihub
https://www.webofscience.com/wos/woscc/full-record/WOS:000779146100004
https://www.webofscience.com/wos/woscc/summary/f0240f82-10dc-4334-a52c-fc1b3ddd547e-017965f705/c44a1530-1ebf-4ce8-899c-7f767e8a7f39-017965f704/relevance/1
https://www.webofscience.com/wos/woscc/summary/9aad0343-3207-4ccc-bffc-2b76ebbe8628-0179658328/86a22084-e197-47db-a4be-1f373afddc71-017965830f/relevance/1
https://www.webofscience.com/wos/woscc/summary/5b1bd317-cdd3-427e-a4aa-f38c621dc2ff-01796579a8/1d3fc4dc-3920-4245-acf0-dbad294f3cc5-017965799f/relevance/1
https://www.webofscience.com/wos/woscc/summary/ef11119b-cb53-4df3-b521-296557b754e0-01796569eb/8f58f9ba-ea95-4c9b-bb0e-12c41d98816b-01796569dd/relevance/1
https://www.webofscience.com/wos/woscc/summary/1974eeb2-0162-4f59-a79a-5fa71363293f-0179654876/5886f0ba-5c07-4075-9d7a-f722948c8ee6-0179656249/relevance/1
https://scholar.google.com/scholar?start=10&q=%22dual+lifestyle%22+fungi+OR+%22facultative+endophyte%22+OR+%22mycorrhizal+AND+saprotrophic%22&hl=en&as_sdt=0,3
https://nph.onlinelibrary.wiley.com/doi/10.1111/j.1469-8137.2009.02987.x
https://www.jstor.org/stable/26928204?seq=1
https://www.jstor.org/stable/26928204
https://arizona-nau.primo.exlibrisgroup.com/openurl/01NAU_INST/01NAU_INST:01NAU?url_ver=Z39.88-2004&url_ctx_fmt=info:ofi/fmt:kev:mtx:ctx&rft.atitle=Some+mycoheterotrophic+orchids+depend+on+carbon+from+dead+wood%3A+novel+evidence+from+a+radiocarbon+approach&rft.aufirst=Kenji&rft.auinit=K&rft.aulast=Suetsugu&rft.date=2020&rft_id=info:doi/10.1111%2Fnph.16409&rft.eissn=1469-8137&rft.epage=1529&rft.genre=article&rft.issn=0028-646X&rft.issue=5&rft.jtitle=NEW+PHYTOLOGIST&rft_val_fmt=info:ofi/fmt:kev:mtx:journal&rft.pages=1519-1529&rfr_id=info:sid/webofscience.com:WOS:WOSCC&rft.spage=1519&rft.stitle=NEW+PHYTOL&rft.volume=227&rft.au=Suetsugu%2C+K&rft.au=Matsubayashi%2C+J&rft.au=Tayasu%2C+I
https://www.webofscience.com/wos/woscc/summary/c3c6c2fd-1e9d-4790-ba6d-2e6b2b10191c-0179654934/fa90c9ce-705e-483c-9330-2fab95378785-0179654877/relevance/1
https://www.webofscience.com/wos/woscc/summary/1974eeb2-0162-4f59-a79a-5fa71363293f-0179654876/fa90c9ce-705e-483c-9330-2fab95378785-0179654877/relevance/1
https://scholar.google.com/scholar?hl=en&as_sdt=0%2C3&q=%22dual+lifestyle%22+fungi+OR+%22facultative+endophyte%22+OR+%22mycorrhizal+AND+saprotrophic%22&btnG=
https://mail.google.com/mail/u/0/?tab=wm#search/francis/QgrcJHsbdwdpbMqZmMRjlqRfxfNhcLVLBGq
https://arizona-nau.primo.exlibrisgroup.com/discovery/search?query=any,contains,nutrient%20cycling%20by%20saprotrophic%20fungi%20in%20terrestrial%20habitats&tab=Everything&search_scope=MyInst_and_CI&vid=01NAU_INST:01NAU&offset=0
https://library.nau.edu/services/search/quick-search-processing.php?choice=Everything&search=nutrient%20cycling%20by%20saprotrophic%20fungi%20in%20terrestrial%20habitats&version=primo-ve&submit=Submit
https://www.pnas.org/doi/epdf/10.1073/pnas.96.15.8534
https://www.jstor.org/stable/1353665?casa_token=l1ugDhVmQ-cAAAAA%3AQj6tdUIPI9S24onNuapoYNszXeKb-h7NoZHMgVm2J9NyDl67z-YgSuWkaDUi9DaZtmBBxEy_8jQBmoA8AmtcNV58FZk65zuEoI7MwZFoiNAXwNkvM0w&seq=1
https://www.jstor.org/stable/1353665?casa_token=l1ugDhVmQ-cAAAAA%3AQj6tdUIPI9S24onNuapoYNszXeKb-h7NoZHMgVm2J9NyDl67z-YgSuWkaDUi9DaZtmBBxEy_8jQBmoA8AmtcNV58FZk65zuEoI7MwZFoiNAXwNkvM0w
https://www.jstor.org/stable/1353665?casa_token=l1ugDhVmQ-cAAAAA:Qj6tdUIPI9S24onNuapoYNszXeKb-h7NoZHMgVm2J9NyDl67z-YgSuWkaDUi9DaZtmBBxEy_8jQBmoA8AmtcNV58FZk65zuEoI7MwZFoiNAXwNkvM0w
https://scholar.google.com/scholar?hl=en&as_sdt=0%2C3&q=hobbie+mycorrhizal+vs+saprotrophic&btnG=
https://www.pnas.org/doi/abs/10.1073/pnas.96.15.8534
https://onlinelibrary.wiley.com/doi/epdf/10.1111/mec.12224
https://link.springer.com/article/10.1007/s13225-013-0240-y
https://www.annualreviews.org/content/journals/10.1146/annurev-ecolsys-012021-114902
https://mail.google.com/mail/u/0/#inbox/QgrcJHsHmbcxmwBGwnmhrqGKJdPtmTtGNSV
https://mail.google.com/mail/u/0/#inbox/QgrcJHsHmbcxmwBGwnmhrqGKJdPtmTtGNSV?compose=GTvVlcSMTFGsdwwZCghBjJQrNRgWtvDMGftcJgGbKSdXSlRgJSpscgZtSbtRwxmKxFzFtLfHDxZdG
https://mail.google.com/mail/u/0/#inbox/QgrcJHsHmbcxmwBGwnmhrqGKJdPtmTtGNSV?compose=new
https://mail.google.com/mail/u/0/#inbox/QgrcJHrhsvKxsrDNpfdJjCKHrLrxFzfhWml
https://mail.google.com/mail/u/0/#inbox/QgrcJHrhsvKxsrDNpfdJjCKHrLrxFzfhWml?compose=GTvVlcSGLPzFWfhbDphKjJPvfwKgsSbBzdvZldXXDPnBtgBpbJwvJhJZnzLJhvfrnwsSjHkKVtbzg
https://mail.google.com/mail/u/0/#inbox/QgrcJHrhsvKxsrDNpfdJjCKHrLrxFzfhWml?compose=new
https://nph.onlinelibrary.wiley.com/doi/epdf/10.1111/nph.14551
https://nph.onlinelibrary.wiley.com/doi/full/10.1111/nph.14551
https://mail.google.com/mail/u/0/?tab=wm#inbox/KtbxLzGDXQPmCHzmDWZhTZqZvBvdpHDbVB?projector=1&messagePartId=0.1
https://mail.google.com/mail/u/0/?tab=wm#inbox/KtbxLzGDXQPmCHzmDWZhTZqZvBvdpHDbVB
https://chatgpt.com/c/68769574-5924-8010-9ad9-0add86bdfdb0
https://www.uni-bayreuth.de/en/press-release/mycoheterotrophic-plants?utm_source=chatgpt.com