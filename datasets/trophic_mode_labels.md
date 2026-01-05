# Trophic Mode Label Definitions

This supplementary file provides the label definitions used in the BioBERT binary classification task for fungal trophic modes.

## Binary Classification Labels

| Label | Code | Definition |
|-------|------|------------|
| **Solo** | 0 | Taxa restricted to a single trophic mode |
| **Dual** | 1 | Taxa reported to occupy more than one trophic mode |

---

## Detailed Label Descriptions

### Solo (Label = 0)
Abstracts classified as "solo" describe fungi that are reported to occupy **a single, specialized trophic mode**. Examples include:

- **Obligate symbionts**: Fungi that require a host organism and cannot complete their life cycle independently (e.g., obligate mycorrhizal fungi, obligate biotrophs)
- **Strict saprotrophs**: Fungi that exclusively decompose dead organic matter
- **Obligate pathogens**: Fungi that can only obtain nutrients from living host tissue
- **Strictly endophytic**: Fungi that live within plant tissues without causing disease and show no other lifestyle

**Example search terms used:**
- "obligate mycorrhizal"
- "strictly endophytic"
- "exclusive saprotroph"

### Dual (Label = 1)
Abstracts classified as "dual" describe fungi that are reported to occupy **multiple trophic modes**, either simultaneously, sequentially, or depending on environmental context. Examples include:

- **Facultative pathogens/saprotrophs**: Fungi that can switch between pathogenic and saprotrophic lifestyles
- **Endophyte-saprotroph continuum**: Fungi that transition from endophytic to saprotrophic phases
- **Lifestyle switching**: Taxa that alter trophic strategy based on host availability or environmental conditions
- **Dual lifestyle fungi**: Explicitly described as occupying two or more ecological roles

**Example search terms used:**
- "dual lifestyle"
- "facultative lifestyle"
- "dual trophic mode"
- "lifestyle switching"
- "endophyte-saprotroph"
- "plant-associated saprotroph"

---

## Underlying Trophic Mode Categories

The following trophic modes are commonly referenced in fungal ecology literature. The "solo" vs "dual" classification reflects whether a taxon is restricted to one or spans multiple of these categories:

| Trophic Mode | Description |
|--------------|-------------|
| **Saprotroph** | Decomposes dead organic matter |
| **Symbiont (Mycorrhizal)** | Forms mutualistic associations with plant roots |
| **Pathogen** | Causes disease in living hosts (plants, animals, fungi) |
| **Endophyte** | Lives within plant tissues without causing apparent disease |
| **Parasite** | Obtains nutrients at the expense of a living host |
| **Lichenized** | Forms symbiotic associations with photobionts (algae/cyanobacteria) |
| **Mycoparasite** | Parasitizes other fungi |

---

## Selection Criteria for Labeling

Abstracts were labeled based on explicit statements in the text:

1. **Unambiguous description required**: The abstract must clearly state the trophic mode(s) of the fungus/fungi discussed
2. **Explicit keywords**: Labels were assigned based on presence of definitive language (e.g., "obligate," "facultative," "dual," "switching")
3. **Conservative exclusion**: Abstracts with ambiguous or implied trophic modes were excluded from the dataset

---

## Reference Databases

This classification scheme is designed to complement existing fungal trait databases:

- **FUNGuild** (Nguyen et al. 2016): Assigns ecological guilds to fungal taxa based on taxonomic identity
- **FungalTraits** (Põlme et al. 2021): Comprehensive database of fungal functional traits including trophic modes

The binary solo/dual classification adds a layer of trophic flexibility information that can enhance these resources by flagging taxa known to exhibit lifestyle plasticity.
